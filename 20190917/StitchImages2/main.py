#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/19

# SIFT特征检测+FLANN特征匹配

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Stitch():
    def work(self):
        print("work")

    def StitchTwo(self, img1, img2):
        top, bot, left, right = 100, 100, 0, 500
        srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        rows0, cols0 = img1.shape[:2]
        rows, cols = srcImg.shape[:2]
        print("0000000=", rows0, cols0)
        print("0000000=", rows, cols)
        img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
        img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d_SIFT().create()
        # find the keypoints and descriptors with SIFT
        # 特征点检测
        kp1, des1 = sift.detectAndCompute(img1gray, None)
        kp2, des2 = sift.detectAndCompute(img2gray, None)
        # FLANN parameters
        # 特征点匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        # 模板
        matchesMask = [[0, 0] for i in range(len(matches))]

        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
        # cv.imshow('match', img3)
        plt.imshow(img3, ), plt.show()

        rows, cols = srcImg.shape[:2]
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                         flags=cv.WARP_INVERSE_MAP)

            for col in range(0, cols):
                if srcImg[:, col].any() and warpImg[:, col].any():
                    left = col
                    break
            for col in range(cols - 1, 0, -1):
                if srcImg[:, col].any() and warpImg[:, col].any():
                    right = col
                    break

            # 目标图应该是大一号的图，笨方法乘以1.5
            offrow = 0 # rows // 4
            offcol = 0 # cols // 4
            res = np.zeros([int(rows*1), int(cols*1), 3], np.uint8)
            print("1111111=", rows, cols, res.shape)
            for row in range(0, rows):
                for col in range(0, cols):
                    if not srcImg[row, col].any():
                        res[row+offrow, col+offcol] = warpImg[row, col]
                    elif not warpImg[row, col].any():
                        res[row+offrow, col+offcol] = srcImg[row, col]
                    else:
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row+offrow, col+offcol] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

            print("2222222=", res.shape)

            # resuce black border
            res = self.ReduceBorder(res)

            # 上采样进行复原
            # res = cv.pyrUp(res)
            # res = cv.pyrUp(res)

            # opencv is bgr, matplotlib is rgb
            res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
            # cv.imshow('res', res)
            cv.imwrite('res.png', res)
            # show the result
            plt.figure()
            plt.imshow(res)
            plt.show()
            return res
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

    # 裁剪黑边
    def AutoReduceBorder(self, img):
        rows, cols = img.shape[:2] # 原图尺寸
        plt.imshow(img, ), plt.show() # 显示原图
        h = 0 # 黑边的高
        w = 0 # 黑边的宽
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 转换成灰度图
        # 垂直方向遍历
        for row in range(0, rows):
            px = gray[row, cols//2]
            if(px == 0):
                h = h + 1
            else:
                break
        # 水平方向遍历
        i = cols - 1
        while i > 0:
            px = gray[rows // 2, i]
            if (px == 0):
                w = w + 1
                i = i - 1
            else:
                break
        print("h = ", h) # 100
        print("w = ", w) # 500
        res = img[h+1:rows, 0:cols-w]
        return res

    def ReduceBorder(self, img):
        rows, cols = img.shape[:2]  # 原图尺寸
        # plt.imshow(img, ), plt.show() # 显示原图
        res = img[101:rows, 0:cols-500]
        # plt.figure(), plt.imshow(res, ), plt.show()  # 显示裁剪图
        return res

    # height = rows - 100
    # width = cols - 500
    # res = np.zeros([width, height, 3], np.uint8)
    # for h in range(0, height):
    #     for w in range(0, width):
    #         print('h:', h, 'w:', w)
    #         res[h, w] = img[h+100, w]


if __name__ == '__main__':
    # img1 = cv.imread('images/3.png')
    # img2 = cv.imread('images/4.png')
    img1 = cv.imread('res_1_2.png')
    img2 = cv.imread('res_3_4.png')
    img1 = cv.resize(img1, (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
    img2 = cv.resize(img2, (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
    # 下采样
    # img1 = cv.pyrDown(img1)
    # img2 = cv.pyrDown(img2)
    # img1 = cv.pyrDown(img1)
    # img2 = cv.pyrDown(img2)
    S = Stitch()
    S.StitchTwo(img1, img2)



