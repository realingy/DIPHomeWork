#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ingy time:2019/9/12

#Panorama Stitcher

import cv2
import numpy as np

class macthing(object):
    def matchIMG(self,im1,im2,kp1,kp2,des1,des2):
        FLANN_INDEX_KDTREE=0
        index_p=dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        searth_p=dict(checks=50)
        flann=cv2.FlannBasedMatcher(index_p,searth_p)
        matches=flann.knnMatch(des1,des2,k=2)
        good =[]
        pts1=[]
        pts2=[]
        for i,(m,n) in enumerate(matches):
            if m.distance<0.6*n.distance:
                good.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        pts1=np.float32(pts1)
        pts2=np.float32(pts2)
        H,mask=cv2.findHomography(pts1,pts2,cv2.RANSAC,0.01)
        pts1_1=pts1[mask.ravel()==1]
        pts2_2=pts2[mask.ravel()==1]
        return pts1_1,pts2_2,H
    def appendimage(self,im1,im2):
        r1=im1.shape[0]
        r2=im2.shape[0]
        if r1<r2:
            img=np.zeros((r2-r1,im1.shape[1]),np.uint8)
            im1_1=np.vstack((im1,img))
            im3=np.hstack((im1_1,im2))
        else:
            img=np.zeros((r1-r2,im2.shape[1]),np.uint8)
            im2_2=np.vstack((im2,img))
            im3=np.hstack((im1,im2_2))
        return im3
#    def panorama_get(self,im1,im2,H):
#        if im1.shape[0]>=im2.shape[0]:
#            result=cv2.warpPerspective(im1,H,(im1.shape[1]+im2.shape[1],im1.shape[0]))
#            result[0:im2.shape[0],0:im2.shape[1]]=im2
#        else:
#            result=cv2.warpPerspective(im1,H,(im1.shape[1]+im2.shape[1],im2.shape[0]))
#            result[0:im2.shape[0],0:im2.shape[1]]=im2
#        return result
    def panorama_get(self,im1,im2,H):
        h1,w1=im1.shape[:2]
        h2,w2=im2.shape[:2]
        pts1=np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2) # 转为3维坐标
        pts2=np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        pts1_=cv2.perspectiveTransform(pts1,H) # H：3*3 矩阵，所以pts1也该为3维坐标
        pts=np.concatenate((pts1_,pts2),axis=0) # 列连接
        # np.min 是行优先
        [xmin,ymin]=np.int32(pts.min(axis=0).ravel()-0.5)
        [xmax,ymax]=np.int32(pts.max(axis=0).ravel()+0.5)
        t=[-xmin,-ymin] # 左加右减
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # 相当于一个向右平移
        result = cv2.warpPerspective(im1, Ht.dot(H), (xmax-xmin, ymax-ymin)) # 最后一个参数是输出图像的宽、高
        result[t[1]:h2+t[1],t[0]:w2+t[0]] = im2
        return result


def work(im1, im2):
    M = macthing()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1=sift.detectAndCompute(im1,None)
    kp2,des2=sift.detectAndCompute(im2,None)
    pts1_1,pts2_2,H=M.matchIMG(im1,im2,kp1,kp2,des1,des2)
    im3=M.appendimage(im1,im2)
    pts2_new=pts2_2.copy()
    for i in range(len(pts2_2)):
        pts2_new[i,0]=pts2_new[i,0]+np.float32(b1.shape[1])
    for i in range(len(pts1_1)):
        cv2.line(im3,tuple(pts1_1[i]),tuple(pts2_new[i]),255,2)
#    cv  cv2\Y\Desktop\45.jpg",result)
    result = M.panorama_get(im1, im2, H)
    return result


if __name__=="__main__":
    # M=macthing()
    # im1_=cv2.imread(r"left.jpg")
    # im2_=cv2.imread(r"right.jpg")
    im1_=cv2.imread(r"left.png")
    im2_=cv2.imread(r"right.png")

    # im1=cv2.cvtColor(im1_,cv2.COLOR_BGR2GRAY)
    # im2=cv2.cvtColor(im2_,cv2.COLOR_BGR2GRAY)

    b1, g1, r1 = cv2.split(im1_)  # 分离函数
    b2, g2, r2 = cv2.split(im2_)  # 分离函数

    b = work(b1, b2)
    g = work(g1, g2)
    r = work(r1, r2)
    cv2.namedWindow("b", cv2.WINDOW_NORMAL)
    cv2.imshow("b", b)
    cv2.namedWindow("g", cv2.WINDOW_NORMAL)
    cv2.imshow("g", g)
    cv2.namedWindow("r", cv2.WINDOW_NORMAL)
    cv2.imshow("r", r)


#     sift=cv2.xfeatures2d.SIFT_create()
#     kp1,des1=sift.detectAndCompute(b1,None)
#     kp2,des2=sift.detectAndCompute(b2,None)
#     pts1_1,pts2_2,H=M.matchIMG(b1,b2,kp1,kp2,des1,des2)
#     im3=M.appendimage(b1,b2)
#     pts2_new=pts2_2.copy()
#     for i in range(len(pts2_2)):
#         pts2_new[i,0]=pts2_new[i,0]+np.float32(b1.shape[1])
#     for i in range(len(pts1_1)):
#         cv2.line(im3,tuple(pts1_1[i]),tuple(pts2_new[i]),255,2)
#    cv  cv2\Y\Desktop\45.jpg",result)
#     result = M.panorama_get(b1, b2, H)

    # result = cv2.merge([b,g,r])
    # cv2.namedWindow("panorma", cv2.WINDOW_NORMAL)
    # cv2.imshow("panorma",result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

