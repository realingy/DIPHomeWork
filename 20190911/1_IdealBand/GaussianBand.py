#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/16

"""
高斯带阻/带通滤波器
"""
import numpy as np

class GaussianBand():
    # 定义函数，高斯带阻/通滤波模板
    def generate(self, src, w, d0, ftype='elimin'):
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


