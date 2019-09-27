#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/27

"""
from ctypes import *
import cv2 as cv

dll = cdll.LoadLibrary('dlltest.dll')
ret = dll.IntAdd(2, 4)
print(ret)

img1 = cv.imread("1.png")
img2 = cv.imread("2.png")

dst = dll.mergeTwoImage(img1, img2)

cv.namedWindow("dst", cv.WINDOW_NORMAL)
cv.imshow("dst", dst)
"""

from ctypes import *
import cv2 as cv

lib = cdll.LoadLibrary('dlltest.dll')

class Merge(object):
    def __init__(self, img):
        lib.Merge_new.argtypes = [c_int,c_int,POINTER(c_ubyte)]
        lib.Merge_new.restype = c_void_p
        lib.Merge_bar.argtypes = [c_void_p]
        lib.Merge_bar.restype = c_void_p
        lib.Merge_foobar.argtypes = [c_void_p,c_int,c_int,POINTER(c_ubyte)]
        lib.Merge_foobar.restype = POINTER(c_int)
        (rows, cols) = (img.shape[0], img.shape[1])
        self.obj = lib.Merge_new(rows, cols,img.ctypes.data_as(POINTER(c_ubyte)))

    def bar(self):
        lib.Merge_bar(self.obj)

    def foobar(self, img):
        (rows, cols) = (img.shape[0], img.shape[1])
        return lib.Merge_foobar(self.obj, rows, cols,img.ctypes.data_as(POINTER(c_ubyte)))

if __name__ == '__main__':
    img = cv.imread('1.png')
    f = Merge(img)
    f.bar()

    # root_path = 'E:/tracking/data/Girl'
    # imglist = scan_image(os.path.join(root_path, 'img'))
    # for imgname in imglist:
    #     img = cv2.imread(os.path.join(root_path, 'img', imgname))
    #     f = Foo(img)
    #     f.bar()
    #     rect = f.foobar(img)
    #     print(rect[0],rect[1])

