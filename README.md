# 简单说明

这是一个用于评估医学图像任务的工具包<br>
medpy相关的评估指标在metrics/binary.py<br>
在medpy包的基础上，我们进行了补充，在metrics/metrics_all.py
包括：<br>
accuracy，precision，recall，specificity，f1_score，dice_coefficient，iou，g_mean，
mae，hausdorff_distance，hausdorff_95，ssim，ncc，psnr，cohen_kappa，log_loss，
fpr，fnr，voe，rvd，sensitivity，jaccard_coefficient，tnr，tpr, 混淆矩阵，ROC，AUC，误分类率，MCC，FDR，NPV，balanced_accuracy, 
mse,MI,NMI,CC, 交叉熵, FID
<br>
我们保留了medpy中处理输入图像的相关部分，在load.py中<br>

## 支持的图像文件格式

医疗格式：<br>
ITK MetaImage (.mha/.raw, .mhd)<br>
NIfTI (.nia, .nii, .nii.gz, .hdr, .img, .img.gz)<br>
Analyze (.hdr/.img, .img.gz)<br>
Nearly Raw Raster Data (Nrrd) (.nrrd, .nhdr)<br>
Medical Imaging NetCDF (MINC) (.mnc, .MNC)<br>
Guys Image Processing Lab (GIPL) (.gipl, .gipl.gz)<br>
<br>显微镜格式：<br>
Medical Research Council (MRC) (.mrc, .rec)<br>
Bio-Rad (.pic, .PIC)<br>
LSM (Zeiss) 显微镜图像 (.tif, .TIF, .tiff, .TIFF, .lsm, .LSM)<br>
Stimulate / Signal Data (SDT) (.sdt)<br>
<br>可视化格式：<br>
VTK 图像 (.vtk)<br>
<br>其他格式：<br>
Portable Network Graphics (PNG) (.png, .PNG)<br>
Joint Photographic Experts Group (JPEG) (.jpg, .JPG, .jpeg, .JPEG)<br>
Tagged Image File Format (TIFF) (.tif, .TIF, .tiff, .TIFF)<br>
Windows bitmap (.bmp, .BMP)<br>
Hierarchical Data Format (HDF5) (.h5 , .hdf5 , .he5)<br>
MSX-DOS Screen-x (.ge4, .ge5)<br>

## 返回类型

图像数据：以 NumPy 数组的形式返回，数据的维度顺序为 x, y, z, c。<br>
<br>头信息：返回 Header 对象，包含图像的元数据。

## 错误处理

如果图像文件不存在，抛出 ImageLoadingError 异常.


## Colab Demo

You can run the demo directly in Google Colab by clicking the link below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16DLfxqDoiTjslmxODYxyvW4yPDtbL4Ew?usp=sharing)

Or use this direct link: https://colab.research.google.com/drive/16DLfxqDoiTjslmxODYxyvW4yPDtbL4Ew?usp=sharing

## Installation
The code requires `python>=3.8`, as well as `pytorch` and `torchvision`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

You can use this package by 

```
pip install git+https://github.com/HanRu1/M.git
```
