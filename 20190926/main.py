#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/19

# SIFT特征检测+FLANN特征匹配

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import time

class Stitch():

    names=[]
    paths=[]
    points=[]
    imageWidth=0
    imageHeight=0

    def __init__(self):
        dir = "images/"
        self.names = os.listdir(dir)
        # print(names)
        self.names.sort()

        for name in self.names:
            self.paths.append("images/" + name)

        img = cv.imread(self.paths[0])
        sp = img.shape
        self.imageHeight = sp[0]
        self.imageWidth = sp[1]

    def work(self):
        length = len(self.names)
        # res1 = self.stitch(0, 2, 400, 300)
        # cv.imwrite('res.png', res1)
        # res1 = self.stitch(0, length//2, 400, 300)
        res1 = self.stitch(0, 8, 400, 300)
        cv.imwrite('res1.png', res1)
        # res2 = self.stitch(length//2, length, 400, 300)
        # cv.imwrite('res2.png', res2)
        # res = self.stitchtwo(res1, res2, 3300, 5500)
        # cv.imwrite('res_0_23.png', res)

        # res1 = cv.imread('1.jpg')
        # res2 = cv.imread('2.jpg')
        # res = self.stitchtwo(res1, res2, 100, 500)

    def stitch(self, start, end, addheight, addwidth):
        img0 = cv.imread(self.paths[start])
        img1 = cv.imread(self.paths[start+1])

        print("==============================Start stitching===============================")
        print("stitching ", self.names[start+1])
        a = time.time()
        res = self.stitchtwo(img0, img1, addheight, addwidth)

        for i in range(start+2, end):
            print("stitching ", self.names[i])
            img = cv.imread(self.paths[i])
            res = self.stitchtwo(res, img, addheight, addwidth)

        print("==============================End stitching===============================")
        print("Totla interval: ", time.time()-a)
        return res

    def stitchtwo(self, img1, img2, addheight, addwidth):
        a = time.time()
        # size matches
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        h = rows1 - rows2
        w = cols1 - cols2
        if(h > 0 or w > 0):
            top, bot, left, right = 0, h, 0, w
            img2 = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        top, bot, left, right = 0, addheight, addwidth, 0
        srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
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

            # sp = testImg.shape
            # self.points.clear()
            # self.points.append(sp[1] - self.imageWidth)
            # self.points.append(0)
            # self.points.append(self.imageWidth)
            # self.points.append(self.imageHeight)
            # roi = testImg[self.points[1]:self.points[3], self.points[0]:self.points[0]+self.points[2]]
            # cv.imwrite("roi.png", roi)
            # calROICorners

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
                        # res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
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
    S.work()
    # img0 = cv.imread('res_0_9.png')
    # img1 = cv.imread('res_10_20.png')
    # S.StitchTwo(img0, img1)



