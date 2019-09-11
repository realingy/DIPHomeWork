#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/11

# -*- coding: utf-8 -*-

"""
理想低通滤波器
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def idealLP(img, w, h):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - int(w/2):crow + int(w/2), ccol - int(h/2):ccol + int(h/2)] = 1
    return mask


# 读取图像
img = cv2.imread('test.bmp', 0)

# 傅里叶变换
# dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft = np.fft.fft2(img)
fshift = np.fft.fftshift(dft)

# 设置低通滤波器
mask = idealLP(img, 60, 60)

# 滤波器和频谱图像乘积
print(fshift.shape)
print(mask.shape)
f = fshift * mask
# print(f.shape, fshift.shape, mask.shape)

# 傅里叶逆变换
ishift = np.fft.ifftshift(f)
d_shift = np.array(np.dstack([ishift.real,ishift.imag]))
iimg = cv2.idft(d_shift)
res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

# 显示原始图像和低通滤波处理图像
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.show()
