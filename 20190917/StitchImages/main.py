#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/18

from imutils import paths
import numpy as np
import imutils
import cv2

from stitch import Stitch

if __name__=="__main__":
    S = Stitch()
    # S.work() # 图像拼接
    src = cv2.imread("res.png")
    S.affine(src) # 仿射处理

