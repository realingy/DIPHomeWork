﻿#include <iostream>  
#include <stdio.h>  
#include "opencv2/core.hpp"  
#include "opencv2/core/utility.hpp"  
#include "opencv2/core/ocl.hpp"  
#include "opencv2/imgcodecs.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/features2d.hpp"  
#include "opencv2/calib3d.hpp"  
#include "opencv2/imgproc.hpp"   
#include "opencv2/xfeatures2d.hpp"  
#include "opencv2/ml.hpp" 
#include <opencv2/core.hpp>
#include "path.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

void CalcCorners(const Mat & H, const Mat & src);
void OptimizeSeam(Mat & img, Mat & wrap, Mat & dst);

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

int main()
{
	Mat left = imread( MediaPath + "SIFT/left.jpg", 1);
	Mat right = imread( MediaPath + "SIFT/right.jpg", 1);

#if 0
	Mat left(left_.rows + 400, left_.cols + 200, CV_8UC3);
	Mat right(right_.rows + 400, right_.cols + 200, CV_8UC3);
	//Mat left();
	//Mat right();

	int borderType = BORDER_REPLICATE;
	RNG rng(12345);
	Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	int addtop = 200;
	int addbottom = 200;
	int addleft = 100;
	int addright = 100;
	copyMakeBorder(left_, left, addtop, addbottom, addleft, addright, borderType, value);
	copyMakeBorder(right_, right, addtop, addbottom, addleft, addright, borderType, value);
#endif

	//imshow("left", left);
	//imshow("right", right);

	Ptr<SIFT> sift;            //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2d
							  //否则即使配置好了还是显示SURF为未声明的标识符  
	sift = SIFT::create(800);

	BFMatcher matcher;         //实例化一个暴力匹配器
	Mat key_left, key_right;
	vector<KeyPoint> key1, key2;
	vector<DMatch> matches;    //DMatch是用来描述匹配好的一对特征点的类，包含这两个点之间的相关信息
							   //比如左图有个特征m，它和右图的特征点n最匹配，这个DMatch就记录它俩最匹配，并且还记录m和n的
							   //特征向量的距离和其他信息，这个距离在后面用来做筛选

	sift->detectAndCompute(right, Mat(), key1, key_left); //输入图像，输入掩码，输入特征点，输出Mat，存放所有特征点的描述向量
	sift->detectAndCompute(left, Mat(), key2, key_right); //这个Mat行数为特征点的个数，列数为每个特征向量的尺寸，SURF是64（维）

	matcher.match(key_right, key_left, matches);             //匹配，数据来源是特征向量，结果存放在DMatch类型里面  

	//sort函数对数据进行升序排列
	sort(matches.begin(), matches.end());     //筛选匹配点，根据match里面特征对的距离从小到大排序
	vector<DMatch> good_matches;
	int ptsPairs = std::min(50, (int)(matches.size() * 0.15));
	cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);//距离最小的50个压入新的DMatch
	}

	Mat outimg; //drawMatches这个函数直接画出摆在一起的图
	drawMatches(right, key1, left, key1, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//imshow("特征匹配效果", outimg);

	//计算图像配准点
	vector<Point2f> imagePoints1, imagePoints2;

	for (int i = 0; i < good_matches.size(); i++)
	{
		imagePoints1.push_back(key1[good_matches[i].trainIdx].pt);
		imagePoints2.push_back(key2[good_matches[i].queryIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	//也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵   

	//计算配准图的四个顶点坐标
	CalcCorners(homo, right);
	cout << "left_top:" << corners.left_top << endl;
	cout << "left_bottom:" << corners.left_bottom << endl;
	cout << "right_top:" << corners.right_top << endl;
	cout << "right_bottom:" << corners.right_bottom << endl;

	//图像配准  
	Mat imageWrap; // , imageTransform2;
	warpPerspective(right, imageWrap, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), right.rows * 1.5)); //透视变换
	//warpPerspective(left, imageTransform2, adjustMat*homo, Size(b.cols*1.3, b.rows*1.8));
	//imshow("旋转效果", imageWrap);
//	imwrite("sift_trans.jpg", imageTransform1);

	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageWrap.cols;  //取最右点的长度为拼接图的长度
	int dst_height = imageWrap.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	imageWrap.copyTo(dst(Rect(0, 0, imageWrap.cols, imageWrap.rows)));
	left.copyTo(dst(Rect(0, 0, left.cols, left.rows)));

	//imshow("拼接效果", dst);
	imwrite(MediaPath + "SIFT/dst.jpg", dst);

	//OptimizeSeam(right, imageWrap, dst);
	//imshow("opm_sift_result", dst);

	waitKey(0);
	return 0;
}

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];
}


//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat & img, Mat & wrap, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  
	double processWidth = img.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img.cols;	//注意，是列数*通道数
	double alpha = 1;		//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = wrap.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}
			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
		}
	}
}




