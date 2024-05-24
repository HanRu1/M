import numpy
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.ndimage import (
    _ni_support,
    binary_erosion,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label,
)
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, matthews_corrcoef)
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from skimage.filters import sobel
from skimage.restoration import estimate_sigma
from sewar.full_ref import vifp

#分割
#accuracy，precision，recall，specificity，f1_score，dice_coefficient，iou，g_mean，
#mae，hausdorff_distance，hausdorff_95，ssim，ncc，psnr，cohen_kappa，log_loss，
#fpr，fnr，voe，rvd，sensitivity，jaccard_coefficient，tnr，tpr
def accuracy(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)

    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)
    tn = numpy.count_nonzero(~result & ~reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        accuracy = (tp + tn) / float(tp + fp + tn + fn)
    except ZeroDivisionError:
        accuracy = 0.0

    return accuracy


def precision(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)

    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def recall(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def specificity(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)
    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    return specificity


def f1_score(result, reference, target_value):
    p = precision(result, reference, target_value)
    r = recall(result, reference, target_value)
    try:
        f1 = 2 * (p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0.0
    return f1


def dice_coefficient(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    tp = numpy.count_nonzero(result & reference)
    try:
        dice = 2 * tp / float(numpy.count_nonzero(result) + numpy.count_nonzero(reference))
    except ZeroDivisionError:
        dice = 0.0
    return dice


def iou(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)
    try:
        iou = intersection / float(union)
    except ZeroDivisionError:
        iou = 0.0
    return iou


def g_mean(result, reference, target_value):
    rec = recall(result, reference, target_value)
    spec = specificity(result, reference, target_value)
    try:
        gmean = numpy.sqrt(rec * spec)
    except ZeroDivisionError:
        gmean = 0.0
    return gmean


def mae(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    return numpy.mean(numpy.abs(result - reference))


def hausdorff_distance(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    # 确保输入为数值类型（0和1）
    result = result.astype(numpy.int32)
    reference = reference.astype(numpy.int32)

    # 获取结果和参考中的前景（即像素值为1）的点集
    result_points = numpy.argwhere(result == 1)
    reference_points = numpy.argwhere(reference == 1)

    # 计算双向的 Hausdorff Distance
    forward_hd = directed_hausdorff(result_points, reference_points)[0]
    backward_hd = directed_hausdorff(reference_points, result_points)[0]

    # Hausdorff Distance 是双向距离的最大值
    hd = max(forward_hd, backward_hd)
    return hd


def hausdorff_95(result, reference, slice_size, target_value):
    """
    计算二值图像的分片 Hausdorff Distance 并合并计算 95th Percentile Hausdorff Distance (95HD)

    参数:
    result (numpy.ndarray): 二值图像的分割结果
    reference (numpy.ndarray): 二值图像的参考（真值）
    slice_size (tuple): 切片的大小，形如 (slice_height, slice_width)
    percentile (float): 百分位数，默认是95

    返回:
    float: 95th Percentile Hausdorff Distance
    """
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    result = result.astype(numpy.int32)
    reference = reference.astype(numpy.int32)
    def calculate_slice_hd(result_slice, reference_slice):
        """
        计算单个切片的 Hausdorff Distance
        """
        result_points = numpy.argwhere(result_slice == 1)
        reference_points = numpy.argwhere(reference_slice == 1)

        if len(result_points) == 0 or len(reference_points) == 0:
            return [], []  # 如果某个切片没有前景点，返回无穷大

        distances_result_to_reference = cdist(result_points, reference_points)
        distances_reference_to_result = cdist(reference_points, result_points)

        min_distances_result_to_reference = numpy.min(distances_result_to_reference, axis=1)
        min_distances_reference_to_result = numpy.min(distances_reference_to_result, axis=1)
        # print(f"min_distances_result_to_reference: {min_distances_result_to_reference}")
        # print(f"min_distances_reference_to_result: {min_distances_reference_to_result}")

        # 分割数组成多个切片
        return min_distances_result_to_reference, min_distances_reference_to_result

    # 分割数组成多个切片
    slice_height, slice_width = slice_size
    h, w = result.shape[:2]
    hd_distances = []

    for i in range(0, h, slice_height):
        for j in range(0, w, slice_width):
            result_slice = result[i:i + slice_height, j:j + slice_width]
            reference_slice = reference[i:i + slice_height, j:j + slice_width]

            if result_slice.size == 0 or reference_slice.size == 0:
                continue

            min_distances_result_to_reference, min_distances_reference_to_result = calculate_slice_hd(result_slice, reference_slice)
            hd_distances.extend(min_distances_result_to_reference)
            hd_distances.extend(min_distances_reference_to_result)

    # 计算第 percentile 百分位数距离
    hd_95 = numpy.percentile(hd_distances, 95)
    return hd_95


def calculate_ssim(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    return ssim(result, reference)


def ncc(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)

    mean_result = numpy.mean(result)
    mean_reference = numpy.mean(reference)

    numerator = numpy.sum((result - mean_result) * (reference - mean_reference))
    denominator = numpy.sqrt(numpy.sum((result - mean_result) ** 2) * numpy.sum((reference - mean_reference) ** 2))

    try:
        ncc_value = numerator / denominator
    except ZeroDivisionError:
        ncc_value = 0.0  # If the denominator is zero, set NCC to 0

    return ncc_value


def calculate_psnr(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    return psnr(reference, result)


def cohen_kappa(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    tp = numpy.count_nonzero(result & reference)
    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)
    fn = numpy.count_nonzero(~result & reference)
    total = float(tp + tn + fp + fn)
    p0 = (tp + tn) / total
    pyes = ((tp + fp) * (tp + fn)) / total**2
    pno = ((tn + fn) * (tn + fp)) / total**2
    pe = pyes + pno
    try:
        kappa = (p0 - pe) / (1 - pe)
    except ZeroDivisionError:
        kappa = 0.0
    return kappa


def log_loss(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    result = numpy.clip(result, 1e-15, 1 - 1e-15)  # To avoid log(0) issue
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    return -numpy.mean(reference * numpy.log(result) + (1 - reference) * numpy.log(1 - result))


def fpr(result, reference, target_value):
    return 1 - specificity(result, reference, target_value)


def fnr(result, reference, target_value):
    return 1 - recall(result, reference, target_value)


def voe(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)
    try:
        voe = 1 - intersection / float(union)
    except ZeroDivisionError:
        voe = 0.0
    return voe


def rvd(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    vol_diff = numpy.count_nonzero(result) - numpy.count_nonzero(reference)
    try:
        rvd = vol_diff / float(numpy.count_nonzero(reference))
    except ZeroDivisionError:
        rvd = 0.0
    return rvd


def sensitivity(result, reference, target_value):
    return recall(result, reference, target_value)


def jaccard_coefficient(result, reference, target_value):
    return iou(result, reference, target_value)


def tnr(result, reference, target_value):
    return specificity(result, reference, target_value)


def tpr(result, reference, target_value):
    return recall(result, reference, target_value)


#分类
#第一次出现的：混淆矩阵，ROC，AUC，误分类率，MCC，FDR，NPV，balanced_accuracy，调和平均
#上面已经有的：accuracy，precision，recall，f1_score，specificity，FPR，FNR，TPR，TNR，
def calculate_confusion_matrix(result, reference, target_value):
    # 确保result和reference是二值数组
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    # 将输入数组展平为一维
    result_flat = result.ravel()
    reference_flat = reference.ravel()

    cm = confusion_matrix(reference_flat, result_flat)
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def roc(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    result_flat = result.ravel()
    reference_flat = reference.ravel()

    fpr, tpr, thresholds = roc_curve(reference_flat, result_flat)
    return fpr, tpr, thresholds


def auc(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    # 将输入数组展平为一维
    result_flat = result.ravel()
    reference_flat = reference.ravel()

    return roc_auc_score(reference_flat, result_flat)


def error_rate(result, reference, target_value):
    return 1 - accuracy(result, reference, target_value)


def mcc(result, reference, target_value):
    # 确保result和reference是二值数组
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    # 将输入数组展平为一维
    result_flat = result.ravel()
    reference_flat = reference.ravel()

    return matthews_corrcoef(reference_flat, result_flat)


def fdr(result, reference, target_value):
    tn, fp, fn, tp = calculate_confusion_matrix(reference, result, target_value)
    try:
        fdr = fp / float(fp + tp)
    except ZeroDivisionError:
        fdr = 0.0
    return fdr


def npv(result, reference, target_value):
    tn, fp, fn, tp = calculate_confusion_matrix(reference, result, target_value)
    try:
        npv = tn / float(tn + fn)
    except ZeroDivisionError:
        npv = 0.0
    return npv


def balanced_accuracy(result, reference, target_value):
    return (tpr(result, reference, target_value) + tnr(result, reference, target_value)) / 2


# def harmonic_mean_accuracy(result, reference):
#     """
#     计算调和平均（Harmonic Mean of Class-wise Accuracy）。
#     """
#     from scipy.stats import hmean
#     unique_labels = numpy.unique(reference)
#     accuracies = [accuracy(result[reference==label], reference[reference==label]) for label in unique_labels]
#     return hmean(accuracies)

#检测
#第一次出现的：无
#上面已经有的：混淆矩阵，precision，recall，f1_score，
#specificity，accuracy，iou，fpr，fnr，dice，sensitivity，fdr，npv，ROC，AUC，tnr

#配准
#第一次出现的：mse,MI,NMI,CC
#上面已经有的：mae，ssim，Dice，Jaccard，Hausdorff，psnr，NCC，


def mse(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    result = result.astype(numpy.int32)
    reference = reference.astype(numpy.int32)

    return numpy.mean((result - reference) ** 2)


def mutual_information(result, reference, target_value):
    # 将输入数组展平为一维
    result_flat = result.ravel()
    reference_flat = reference.ravel()

    # 确保result和reference是二值数组（0和1）
    result_flat = numpy.atleast_1d(result_flat == target_value)
    reference_flat = numpy.atleast_1d(reference_flat == target_value)

    return mutual_info_score(reference_flat, result_flat)

def normalized_mutual_information(result, reference, target_value):
    # 将输入数组展平为一维
    result_flat = result.ravel()
    reference_flat = reference.ravel()

    result = numpy.atleast_1d(result_flat == target_value)
    reference = numpy.atleast_1d(reference_flat == target_value)
    try:
        nmi = normalized_mutual_info_score(result, reference)
    except ZeroDivisionError:
        nmi = 0.0
    return nmi


def correlation_coefficient(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)
    # 计算均值
    mean_x = numpy.mean(result)
    mean_y = numpy.mean(reference)
    # 计算分子和分母
    numerator = numpy.sum((result - mean_x) * (reference - mean_y))
    denominator = numpy.sqrt(numpy.sum((result - mean_x) ** 2) * numpy.sum((reference - mean_y) ** 2))
    try:
        corr_coef = numerator / denominator
    except ZeroDivisionError:
        corr_coef = 0.0
    return corr_coef


#融合
#第一次出现的：交叉熵,平均梯度,空间频率,标准偏差,噪声估计
#上面已经有的：mse，psnr，ssim，cc


def cross_entropy(result, reference, target_value):
    result = numpy.atleast_1d(result == target_value)
    reference = numpy.atleast_1d(reference == target_value)

    result = numpy.clip(result, 1e-15, 1 - 1e-15)

    try:
        entropy = -numpy.sum(reference * numpy.log(result))
    except ZeroDivisionError:
        entropy = 0.0
    return entropy


def calculate_average_gradient(image):
    #平均梯度
    try:
        dx = sobel(image, axis=0)
        dy = sobel(image, axis=1)
        dz = sobel(image, axis=2)
        gradient_magnitude = numpy.sqrt(dx**2 + dy**2 + dz**2)
        return numpy.mean(gradient_magnitude)
    except Exception as e:
        return 0.0


def calculate_spatial_frequency(image):
    #空间频率
    try:
        dx = numpy.var(sobel(image, axis=0))
        dy = numpy.var(sobel(image, axis=1))
        return numpy.sqrt(dx + dy)
    except Exception as e:
        return 0.0


def calculate_standard_deviation(image):
    #标准偏差
    try:
        return numpy.std(image)
    except Exception as e:
        return 0.0


def estimate_noise(image):
    #噪声估计
    try:
        return estimate_sigma(image, multichannel=False, average_sigmas=True)
    except Exception as e:
        return 0.0


#重建
#第一次出现的：计算体积相似性,计算点到点距离,VIF
#上面已经有的：mse，psnr，ssim，Dice，Jaccard，Hausdorff，
def calculate_volume_similarity(result, reference):
    # 确保输入数组为浮点型，以避免溢出
    result = numpy.asarray(result, dtype=numpy.float64)
    reference = numpy.asarray(reference, dtype=numpy.float64)

    # 计算体积
    volA = numpy.sum(result)
    volB = numpy.sum(reference)

    # 处理异常情况
    try:
        if volA + volB == 0:
            volume_similarity = 0.0
        else:
            volume_similarity = 1 - numpy.abs(volA - volB) / (volA + volB)
    except ZeroDivisionError:
        volume_similarity = 0.0  # 当volA和volB都是0时，返回0.0

    return volume_similarity


def point_to_point_distance(result, reference):
    # 确保输入是NumPy数组
    result = numpy.atleast_1d(result.astype(numpy.float32))
    reference = numpy.atleast_1d(reference.astype(numpy.float32))

    # 计算逐点的欧氏距离
    distances = numpy.sqrt(numpy.sum((result - reference) ** 2, axis=-1))

    # 返回距离数组
    return distances


# def calculate_vif(result, reference):
#     # 将输入数据转换为浮点型并确保至少是3维数组
#     result = numpy.atleast_3d(result.astype(numpy.float64))
#     reference = numpy.atleast_3d(reference.astype(numpy.float64))
#
#     # 检查并清理无效值
#     result[numpy.isnan(result) | numpy.isinf(result)] = 0
#     reference[numpy.isnan(reference) | numpy.isinf(reference)] = 0
#
#     try:
#         vif = vifp(reference, result)
#     except Exception as e:
#         print(f"Error calculating VIF: {e}")
#         vif = 0.0
#
#     return vif
