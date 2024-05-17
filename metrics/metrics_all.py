import numpy
from scipy.spatial.distance import directed_hausdorff
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
def accuracy(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))

    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)
    tn = numpy.count_nonzero(~result & ~reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        accuracy = (tp + tn) / float(tp + fp + tn + fn)
    except ZeroDivisionError:
        accuracy = 0.0

    return accuracy


def precision(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))

    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def recall(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def specificity(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)
    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    return specificity


def f1_score(result, reference):
    p = precision(result, reference)
    r = recall(result, reference)
    try:
        f1 = 2 * (p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0.0
    return f1


def dice_coefficient(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    tp = numpy.count_nonzero(result & reference)
    try:
        dice = 2 * tp / float(numpy.count_nonzero(result) + numpy.count_nonzero(reference))
    except ZeroDivisionError:
        dice = 0.0
    return dice


def iou(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)
    try:
        iou = intersection / float(union)
    except ZeroDivisionError:
        iou = 0.0
    return iou


def g_mean(result, reference):
    rec = recall(result, reference)
    spec = specificity(result, reference)
    try:
        gmean = numpy.sqrt(rec * spec)
    except ZeroDivisionError:
        gmean = 0.0
    return gmean


def mae(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.float32))
    reference = numpy.atleast_1d(reference.astype(numpy.float32))
    return numpy.mean(numpy.abs(result - reference))


def hausdorff_distance(result, reference):
    # Using scipy for directed Hausdorff distance, assuming 2D arrays
    return max(directed_hausdorff(result, reference)[0], directed_hausdorff(reference, result)[0])


def hausdorff_95(result, reference):
    # Calculating 95th percentile of the Hausdorff distance
    distances = []
    for r in result:
        distances.append(numpy.min(numpy.linalg.norm(reference - r, axis=1)))
    for r in reference:
        distances.append(numpy.min(numpy.linalg.norm(result - r, axis=1)))
    return numpy.percentile(distances, 95)


def calculate_ssim(result, reference):
    return ssim(result, reference)


def ncc(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.float64))
    reference = numpy.atleast_1d(reference.astype(numpy.float64))

    mean_result = numpy.mean(result)
    mean_reference = numpy.mean(reference)

    numerator = numpy.sum((result - mean_result) * (reference - mean_reference))
    denominator = numpy.sqrt(numpy.sum((result - mean_result) ** 2) * numpy.sum((reference - mean_reference) ** 2))

    try:
        ncc_value = numerator / denominator
    except ZeroDivisionError:
        ncc_value = 0.0  # If the denominator is zero, set NCC to 0

    return ncc_value


def calculate_psnr(result, reference):
    return psnr(reference, result)


def cohen_kappa(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
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


def log_loss(result, reference):
    result = numpy.clip(result, 1e-15, 1 - 1e-15)  # To avoid log(0) issue
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    return -numpy.mean(reference * numpy.log(result) + (1 - reference) * numpy.log(1 - result))


def fpr(result, reference):
    return 1 - specificity(result, reference)


def fnr(result, reference):
    return 1 - recall(result, reference)


def voe(result, reference):
    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)
    try:
        voe = 1 - intersection / float(union)
    except ZeroDivisionError:
        voe = 0.0
    return voe


def rvd(result, reference):
    vol_diff = numpy.count_nonzero(result) - numpy.count_nonzero(reference)
    try:
        rvd = vol_diff / float(numpy.count_nonzero(reference))
    except ZeroDivisionError:
        rvd = 0.0
    return rvd


def sensitivity(result, reference):
    return recall(result, reference)


def jaccard_coefficient(result, reference):
    return iou(result, reference)


def tnr(result, reference):
    return specificity(result, reference)


def tpr(result, reference):
    return recall(result, reference)


#分类
#第一次出现的：混淆矩阵，ROC，AUC，误分类率，MCC，FDR，NPV，balanced_accuracy，调和平均
#上面已经有的：accuracy，precision，recall，f1_score，specificity，FPR，FNR，TPR，TNR，
def calculate_confusion_matrix(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    return confusion_matrix(reference, result)


def roc(result, reference):
    fpr, tpr, thresholds = roc_curve(reference, result)
    return fpr, tpr, thresholds


def auc(result, reference):
    return roc_auc_score(reference, result)


def error_rate(result, reference):
    return 1 - accuracy(result, reference)


def mcc(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    return matthews_corrcoef(reference, result)


def fdr(result, reference):
    tn, fp, fn, tp = calculate_confusion_matrix(reference, result).ravel()
    try:
        fdr = fp / float(fp + tp)
    except ZeroDivisionError:
        fdr = 0.0
    return fdr


def npv(result, reference):
    tn, fp, fn, tp = calculate_confusion_matrix(reference, result).ravel()
    try:
        npv = tn / float(tn + fn)
    except ZeroDivisionError:
        npv = 0.0
    return npv


def balanced_accuracy(result, reference):
    return (tpr(result, reference) + tnr(result, reference)) / 2


def harmonic_mean_accuracy(result, reference):
    """
    计算调和平均（Harmonic Mean of Class-wise Accuracy）。
    """
    from scipy.stats import hmean
    unique_labels = numpy.unique(reference)
    accuracies = [accuracy(result[reference==label], reference[reference==label]) for label in unique_labels]
    return hmean(accuracies)

#检测
#第一次出现的：无
#上面已经有的：混淆矩阵，precision，recall，f1_score，
#specificity，accuracy，iou，fpr，fnr，dice，sensitivity，fdr，npv，ROC，AUC，tnr

#配准
#第一次出现的：mse,MI,NMI,CC
#上面已经有的：mae，ssim，Dice，Jaccard，Hausdorff，psnr，NCC，


def mse(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.float32))
    reference = numpy.atleast_1d(reference.astype(numpy.float32))

    return numpy.mean((result - reference) ** 2)


def mutual_information(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    return mutual_info_score(result, reference)


def normalized_mutual_information(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    try:
        nmi = normalized_mutual_info_score(result, reference)
    except ZeroDivisionError:
        nmi = 0.0
    return nmi


def correlation_coefficient(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.float32))
    reference = numpy.atleast_1d(reference.astype(numpy.float32))
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


def cross_entropy(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.float32))
    reference = numpy.atleast_1d(reference.astype(numpy.float32))

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
    # 计算体积相似性（Volume Similarity）
    volA = numpy.sum(result)
    volB = numpy.sum(reference)
    try:
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


def calculate_vif(result, reference):
    result = numpy.atleast_3d(result.astype(numpy.float64))
    reference = numpy.atleast_3d(reference.astype(numpy.float64))
    try:
        vif = vifp(reference, result)
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        vif = 0.0
    return vif
