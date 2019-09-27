#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/19

# SIFT特征检测+FLANN特征匹配

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import time

class Rect():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

    def __repr__(self):
        """
        定义字节输出
        """
        return str(int(self.x)) + "," + str(int(self.y)) + "," + str(int(self.width)) + "," + str(int(self.height))

class Corner():
    def __init__(self):
        self.left_top_x = 0
        self.left_top_y = 0
        self.right_top_x = 0
        self.right_top_y = 0
        self.left_bottom_x = 0
        self.left_bottom_y = 0
        self.right_bottom_x = 0
        self.right_bottom_y = 0

    def __repr__(self):
        """
        定义字节输出
        """
        return "left_top: (" + str(int(self.left_top_x)) +","+ str(int(self.left_top_y)) + "),right_top: ("+ str(int(self.right_top_x)) +","+ str(int(self.right_top_y)) \
                + "),left_bottom: (" + str(int(self.left_bottom_x)) +","+ str(int(self.left_bottom_y)) + "),right_bottom: (" +  str(int(self.right_bottom_x)) +","+ str(int(self.right_bottom_y)) +")"

roi = Rect()
corner = Corner()

class Stitch():
    names=[]
    paths=[]
    images=[]

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

    def work(self):
        print("==============================Start stitching===============================")
        a = time.time()

        img0 = self.images[0]
        img1 = self.images[1]

        print("stitching ", self.names[1], "")
        dst = self.stitchtwo(img0, img1)
        self.updateROI()

        length = len(self.names)
        for i in range(2, length):
            print("stitching ", self.names[i])
            dst = self.stitch(dst, self.images[i])

        cv.rectangle(dst, (roi.x, roi.y), (roi.x+roi.width, roi.y+roi.height), (0, 0, 255), 2, 2, 0)

        cv.namedWindow("dst", cv.WINDOW_NORMAL)
        cv.imshow("dst", dst)
        cv.imwrite("res.png", dst)

        print("==============================End stitching===============================")
        print("Totla interval: ", time.time()-a)

        cv.waitKey(0)

    def stitch(self, img1, img2):
        a1 = roi.y
        a2 = roi.y+roi.height
        b1 = roi.x
        b2 = roi.x + roi.width
        img1roi = img1[a1:a2,b1:b2]

        temp = self.stitchtwo(img1roi, img2)

        height, width = temp.shape[:2]
        addwidth = width - roi.width
        addheight = height - roi.height
        dst = cv.copyMakeBorder(img1, 0, addheight, addwidth, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))

        dst[roi.y:roi.y+height,roi.x:roi.x+width] = temp

        self.updateROI()

        return dst

    def stitchtwo(self, img1, img2):
        a = time.time()
        # size matches
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        h = rows1 - rows2
        w = cols1 - cols2
        if(h > 0 or w > 0):
            top, bot, left, right = 0, h, w, 0
            img2 = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        top, bot, left, right = 0, 500, 400, 0
        srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
        img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d_SIFT().create(5000)
        # sift = cv.xfeatures2d_SIFT().create(7000)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1gray, None)
        kp2, des2 = sift.detectAndCompute(img2gray, None)
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

            warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                         flags=cv.WARP_INVERSE_MAP)

            self.calROICorners(warpImg)

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

            # opencv is bgr, matplotlib is rgb
            # res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
            # show the result
            # plt.figure(), plt.imshow(res), plt.show()
            print("interval3: ", time.time() - a)
            return res
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

    def calROICorners(self, img):
        rows, cols = img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 左上角
        find = False
        for row in range(0, rows):
            for col in range(0, cols):
                if gray[row, col]:
                    corner.left_top_x = col
                    corner.left_top_y = row
                    find = True
                    break
            if find == True:
                break

        # 左下角
        find = False
        for col in range(0, cols):
            for row in range(0, rows):
                if gray[row, col]:
                    corner.left_bottom_x = col
                    corner.left_bottom_y = row
                    find = True
                    break
            if find == True:
                break

        # 右上角
        find = False
        for col in range(cols-1, 0, -1):
            for row in range(0, rows):
                if gray[row, col]:
                    corner.right_top_x = col
                    corner.right_top_y = row
                    find = True
                    break
            if find == True:
                break

        # 右下角
        find = False
        for row in range(rows-1, 0, -1):
            for col in range(cols-1, 0, -1):
                if gray[row, col]:
                    corner.right_bottom_x = col
                    corner.right_bottom_y = row
                    find = True
                    break
            if find == True:
                break

    def updateROI(self):
        startx = min(corner.left_top_x, corner.left_bottom_x)
        starty = min(corner.left_top_y, corner.right_top_y)
        endx = max(corner.right_top_x, corner.right_bottom_x)
        endy = max(corner.left_bottom_y, corner.right_bottom_y)
        roi.width = int(endx - startx)
        roi.height = int(endy - starty)
        roi.x += startx
        roi.y += starty



if __name__ == '__main__':
    S = Stitch()
    S.work()


