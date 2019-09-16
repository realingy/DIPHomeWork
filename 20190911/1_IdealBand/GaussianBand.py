#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/16

"""
高斯带阻/带通滤波器
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 定义函数，高斯带阻/通滤波模板
def GaussianBand(src, w, d0, ftype='elimin'):
    template = np.zeros(src.shape, dtype=np.float32)  # 构建滤波器
    r, c = src.shape
    for i in np.arange(r):
        for j in np.arange(c):
            distance = np.sqrt((i - r / 2) ** 2 + (j - c / 2) ** 2)
            temp = ((distance**2 - d0**2)/(distance*w+0.00000001))**2
            template[i, j] = 1 - np.exp(-0.5 * temp)
    if ftype == 'pass':
        template = 1 - template
    return template


src = cv2.imread('test.bmp', 0)

# 快速傅里叶变换算法得到频率分布
f = np.fft.fft2(src)

# 默认结果中心点位置是在左上角
# 调用fftshift()函数转移到中间位置
fshift = np.fft.fftshift(f)

# fft结果是复数, 其绝对值结果是振幅
fsrc = np.log(np.abs(fshift))

# 展示结果
plt.subplot(221), plt.imshow(src, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(222), plt.imshow(fsrc, 'gray'), plt.title('Fourier Image')
plt.axis('off')

# 生成带阻滤波器
mask = GaussianBand(fshift, 10, 0)

fshiftdst = mask * fshift

fdst = np.log(np.abs(fshiftdst))

plt.subplot(223), plt.imshow(fdst, 'gray'), plt.title('IdealBand Fourier Image')
plt.axis('off')

# 傅里叶逆变换
idst = np.fft.ifftshift(fshiftdst)
dst = np.fft.ifft2(idst)
dst = np.abs(dst)

# cv2.imwrite('dst.bmp', dst)

plt.subplot(224), plt.imshow(dst, 'gray'), plt.title('Inverse Fourier Fourier')
plt.axis('off')

plt.show()



