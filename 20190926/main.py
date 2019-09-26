#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/19

# SIFT特征检测+FLANN特征匹配

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import time

roi = []

class Stitch():

    names=[]
    paths=[]
    images=[]
    imageWidth=0
    imageHeight=0

    def __init__(self):
        dir = "images/"

        self.names = os.listdir(dir)
        # print(names)
        self.names.sort()

        for name in self.names:
            self.paths.append("images/" + name)

        for path in self.paths:
            img = cv.imread(path)
            self.images.append(img)

        h, w = self.images[0].shape[:2]
        self.imageHeight = h
        self.imageWidth = w
        print("h: ", self.imageHeight, ",w: ", self.imageWidth)

    def updateROI(self):
        print("11111")

    def work(self):
        print("==============================Start stitching===============================")
        a = time.time()

        img0 = self.images[0]
        img1 = self.images[1]

        print("stitching \"", self.names[1])
        dst = self.stitchtwo(img0, img1)

        img2 = self.images[2]
        print("stitching \"", self.names[2])
        dst = self.stitchtwo(dst, img2)
        self.updateROI()

        length = len(self.names)

        for i in range(3, length):
            print("stitching ", self.names[i])
            dst = self.stitch(dst, self.images[i])
            # self.updateROI()

        # // rectangle(dst, cvPoint(roi.x, roi.y), cvPoint(roi.x+roi.width, roi.y+roi.height), Scalar(0, 0, 255), 2, 2, 0);

        cv.namedWindow("拼接效果", cv.WINDOW_NORMAL)
        cv.imshow("拼接效果", dst)
        # cv.imwrite("res.png", dst)

        print("==============================End stitching===============================")
        print("Totla interval: ", time.time()-a)

        # res1 = self.stitch(0, 2, 400, 300)
        # cv.imwrite('res.png', res1)
        # res1 = self.stitch(0, length//2, 400, 300)
        # res1 = self.stitch(0, 8, 400, 300)
        # cv.imwrite('res1.png', res1)
        # res2 = self.stitch(length//2, length, 400, 300)
        # cv.imwrite('res2.png', res2)
        # res = self.stitchtwo(res1, res2, 3300, 5500)
        # cv.imwrite('res_0_23.png', res)

        # res1 = cv.imread('1.jpg')
        # res2 = cv.imread('2.jpg')
        # res = self.stitchtwo(res1, res2, 100, 500)


    def stitch(self, img1, img2):
        img1roi = img1(self.roi)

        temp = self.stitchtwo(img1roi, img2)

        rows, cols = temp.shape[:2]
        addwidth = cols - self.roi.width
        addheight = rows - self.roi.height
        dst = cv.copyMakeBorder(img1, 0, addheight, addwidth, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))

        # copyto
        # temp.copyTo(dst(Rect(roi.x, roi.y, temp.cols, temp.rows)))

        self.updateROI()

        return dst

    def stitchtwo(self, img1, img2):
        a = time.time()
        # size matches
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        h = rows1 - rows2
        w = cols1 - cols2
        # if(h > 0 or w > 0):
        #     top, bot, left, right = 0, h, w, 0
        #     img2 = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        top, bot, left, right = 0, 400, 300, 0
        srcImg = cv.copyMakeBorder(img1, top, bot+h, left+w, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
        img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
        # img1grayROI = img1gray[0:rows1, w: ]
        # img2grayROI = img2gray[0:rows2, w: ]
        # sift = cv.xfeatures2d_SIFT().create(5000)
        sift = cv.xfeatures2d_SIFT().create(7000)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1gray, None)
        kp2, des2 = sift.detectAndCompute(img2gray, None)
        print("interval1: ", time.time() - a)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            # if m.distance < 0.7 * n.distance:
            if m.distance < 0.5 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                matchesMask[i] = [1, 0]

        # draw matches
        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=(255, 0, 0),
        #                    matchesMask=matchesMask,
        #                    flags=0)
        # img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
        # plt.imshow(img3, ), plt.show()

        rows, cols = srcImg.shape[:2]
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            pots = []
            height, width = testImg.shape[:2]
            pots.append(width - self.imageWidth)
            pots.append(0)
            pots.append(self.imageWidth)
            pots.append(self.imageHeight)
            self.calROICorners(np.array(M), pots)

            warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                         flags=cv.WARP_INVERSE_MAP)

            # for col in range(0, cols):
            #     if srcImg[:, col].any() and warpImg[:, col].any():
            #         left = col
            #         break
            # for col in range(cols - 1, 0, -1):
            #     if srcImg[:, col].any() and warpImg[:, col].any():
            #         right = col
            #         break

            print("interval2: ", time.time() - a)

            res = np.zeros([rows, cols, 3], np.uint8)

            for row in range(0, rows):
                for col in range(0, cols):
                    """
                    if not srcImg[row, col].any():
                        res[row, col] = warpImg[row, col]
                    elif not warpImg[row, col].any():
                        res[row, col] = srcImg[row, col]
                    else:
                        # srcImgLen = float(abs(col - left))
                        # testImgLen = float(abs(col - right))
                        # alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
                        res[row, col] = warpImg[row, col]
                    """

                    if warpImg[row, col, 0]:
                        res[row, col] = warpImg[row, col]
                    else:
                        res[row, col] = srcImg[row, col]

            # resuce black border
            # res = self.ReduceBorder(res)

            # opencv is bgr, matplotlib is rgb
            # res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
            # show the result
            # plt.figure(), plt.imshow(res), plt.show()
            print("interval3: ", time.time() - a)
            return res
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

    def calROICorners(self, H, pots):

        # 左上角
        v2 = {pots[0], pots[1], 1} # 左上角
        # v1 = [3] # 变换后的坐标值

        V2 = np.array(v2).reshape(3, 1)
        # V1 = np.array(v1) # 列向量
        v1 = H * V2

        left_top_x = v1[0] / v1[2]
        left_top_y = v1[1] / v1[2]

        # 右上角
        v2 = {pots[0] + pots[2], pots[1], 1}
        V2 = Mat(3, 1, CV_64FC1, v2)  # 列向量
        V1 = Mat(3, 1, CV_64FC1, v1)  # 列向量
        V1 = H * V2

        right_top_x = v1[0] / v1[2]
        right_top_y = v1[1] / v1[2]

        # 左下角
        v2 = { pots[0], pots[1] + pots[3], 1}
        V2 = Mat(3, 1, CV_64FC1, v2)  # 列向量
        V1 = Mat(3, 1, CV_64FC1, v1)  # 列向量
        V1 = H * V2

        left_bottom_x = v1[0] / v1[2]
        left_bottom_y = v1[1] / v1[2]

        # 右下角
        v2 = {pots[0] + pots[2], pots[1] + pots[3], 1}
        V2 = Mat(3, 1, CV_64FC1, v2)  # 列向量
        V1 = Mat(3, 1, CV_64FC1, v1)  # 列向量
        V1 = H * V2

        right_bottom_x = v1[0] / v1[2]
        right_bottom_y = v1[1] / v1[2]


    def AutoReduceBorder(self, img):
        rows, cols = img.shape[:2]
        plt.imshow(img, ), plt.show()
        h = 0
        w = 0
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for row in range(0, rows):
            px = gray[row, cols//2]
            if(px == 0):
                h = h + 1
            else:
                break
        i = cols - 1
        while i > 0:
            px = gray[rows // 2, i]
            if (px == 0):
                w = w + 1
                i = i - 1
            else:
                break
        print("h = ", h)
        print("w = ", w)
        res = img[h+1:rows, 0:cols-w]
        return res

    def ReduceBorder(self, img):
        rows, cols = img.shape[:2]
        res = img[101:rows, 0:cols-500]
        return res


if __name__ == '__main__':
    S = Stitch()
    # S.work()
    # img0 = cv.imread('res_0_9.png')
    # img1 = cv.imread('res_10_20.png')
    # S.StitchTwo(img0, img1)



