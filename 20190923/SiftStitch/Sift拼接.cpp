#include <iostream>  
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
#include <ctime>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

void CalcCorners(const Mat & H, const Mat & src);
void CalcROICorners(const Mat& H, const Rect & roi);
Mat stitchTwo(Mat & img1, Mat & img2);
Mat doStitchTwo(Mat & img1, Mat & img2);
vector<Mat> getFiles(cv::String dir);
void timeCounter(time_t start);
inline void updateROI();

Rect roi(0, 0, 0, 0);
int g_width;
int g_height;

#define BORDERWIDTH 500
#define BORDERHEIGHT 50

vector<cv::String> paths;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;
four_corners_t cornersroi;
string dir = "images";

void stitch()
{
	cout << "stitch start!\n";
	time_t begin = clock();

	vector<Mat> images = getFiles(dir);

	Mat img0 = images[0];
	Mat img1 = images[1];
	g_height = img1.rows;
	g_width = img1.cols;
	cout << "stitching \"" << paths[1] << "\" ";
	Mat dst = doStitchTwo(img0, img1);

	Mat img2 = images[2];
	cout << "stitching \"" << paths[2] << "\" ";
	dst = doStitchTwo(dst, img2);
	updateROI();

#if 1
	int count = images.size();
	for (int i = 3; i < count; i++)
	{
		cout << "stitching \"" << paths[i] << "\" ";
		//dst = doStitchTwo(dst, images[i]);
		//updateROI();
		dst = stitchTwo(dst, images[i]);
	}
#endif

	//rectangle(dst, cvPoint(roi.x, roi.y), cvPoint(roi.x+roi.width, roi.y+roi.height), Scalar(0, 0, 255), 2, 2, 0);

	namedWindow("拼接效果", WINDOW_NORMAL);
	imshow("拼接效果", dst);
	imwrite("res.png", dst);

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "stitching end, total interval: " << interval << endl;

	waitKey(0);
}

int main()
{
	stitch();

	return 0;
}

Mat stitchTwo(Mat & img1, Mat & img2)
{
	Mat img1roi = img1(roi);

	Mat temp = doStitchTwo(img1roi, img2);

	Mat dst;
	int addwidth = temp.cols - roi.width;
	int addheight = temp.rows - roi.height;
	copyMakeBorder(img1, dst, 0, addheight, addwidth, 0, 0, Scalar(0, 0, 0));

	temp.copyTo(dst(Rect(roi.x, roi.y, temp.cols, temp.rows)));

	//update ROI
	updateROI();

	return dst;
}

Mat doStitchTwo(Mat & img1, Mat & img2)
{
	time_t begin = clock();

	Mat imageSrc;
	Mat imageMatch;

	// size matches
	int width1 = img1.cols;
	int height1 = img1.rows;
	int width2 = img2.cols;
	int height2 = img2.rows;

	int addh = height1 - height2;
	int addw = width1 - width2;
	//cout << "addw: " << addw << "addh: " << addh << endl;

	// make border
	int addtop = 0;
	int addbottom = BORDERHEIGHT;
	int addleft = BORDERWIDTH; //小于420出现拼接模糊
	int addright = 0;
	//copyMakeBorder(img2, imageMatch, addh, addbottom , addleft, addw, 0, Scalar(0, 0, 0));
	copyMakeBorder(img2, imageMatch, addtop, addbottom + addh, addleft+addw, addright, 0, Scalar(0, 0, 0));
	int h = imageMatch.rows * 0.2;
	copyMakeBorder(img1, imageSrc, addtop, addbottom + h, addleft, addright, 0, Scalar(0, 0, 0));

	Ptr<SIFT> sift; //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2, 否则即使配置好了还是显示SIFT为未声明的标识符  
	sift = SIFT::create(5000);

	BFMatcher matcher; //实例化一个暴力匹配器
	Mat key_left, key_right;
	vector<KeyPoint> key1, key2;
	vector<DMatch> matches;    //DMatch是用来描述匹配好的一对特征点的类，包含这两个点之间的相关信息
							   //比如左图有个特征m，它和右图的特征点n最匹配，这个DMatch就记录它俩最匹配，并且还记录m和n的
							   //特征向量的距离和其他信息，这个距离在后面用来做筛选

	//timeCounter(begin);

	sift->detectAndCompute(imageMatch, Mat(), key1, key_left); //输入图像，输入掩码，输入特征点，输出Mat，存放所有特征点的描述向量
	sift->detectAndCompute(imageSrc, Mat(), key2, key_right); //这个Mat行数为特征点的个数，列数为每个特征向量的尺寸，SURF是64（维）

	//drawKeypoints(imageSrc, key1, imageSrc);//画出特征点

	matcher.match(key_right, key_left, matches);             //匹配，数据来源是特征向量，结果存放在DMatch类型里面  

	//sort函数对数据进行升序排列
	sort(matches.begin(), matches.end());     //筛选匹配点，根据match里面特征对的距离从小到大排序
	vector<DMatch> good_matches;
	int ptsPairs = std::min(2000, (int)(matches.size()));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]); //距离最小的500个压入新的DMatch
	}

	//Mat outimg; //drawMatches这个函数直接画出摆在一起的图
	//drawMatches(imageMatch, key1, imageSrc, key2, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//namedWindow("特征匹配效果", WINDOW_NORMAL);
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
	//Mat homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	// cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵   

	//计算配准图的四个顶点坐标
	CalcCorners(homo, imageMatch);

	Rect roi = Rect(imageMatch.cols - g_width, 0, g_width, g_height);
	CalcROICorners(homo, roi);

	//图像配准
	Mat imageWrap; // , imageTransform2;
	warpPerspective(imageMatch, imageWrap, homo, Size(imageMatch.cols, imageMatch.rows+h)); //透视变换
	//rectangle(imageWrap, cvPoint(cornersroi.left_bottom.x, cornersroi.left_top.y), cvPoint(cornersroi.right_top.x , cornersroi.right_bottom.y), Scalar(0, 0, 255), 1, 1, 0);


	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageWrap.cols;
	int dst_height = imageWrap.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	for (int i = 1; i < dst_height; ++i) {
		for (int j = 1; j < dst_width; ++j) {
			/*
			if(imageWrap.at<Vec3b>(i, j)[0] != 0)
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j);
			else
				dst.at<Vec3b>(i, j) = imageSrc.at<Vec3b>(i, j);
			*/
			if(imageSrc.at<Vec3b>(i, j)[0] != 0 && imageWrap.at<Vec3b>(i, j)[0] == 0)
				dst.at<Vec3b>(i, j) = imageSrc.at<Vec3b>(i, j);
			else if(imageSrc.at<Vec3b>(i, j)[0] == 0 && imageWrap.at<Vec3b>(i, j)[0] != 0)
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j);
			else
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j) * 0.6 + imageSrc.at<Vec3b>(i, j) * 0.4;
		}
	}

	//timeCounter(begin);
	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "interval: " << interval << endl;

	return dst;
}

inline void updateROI()
{
	int startx = min(cornersroi.left_top.x, cornersroi.left_bottom.x);
	int starty = min(cornersroi.left_top.y, cornersroi.right_top.y);
	int endx = max(cornersroi.right_top.x, cornersroi.right_bottom.x);
	int endy = max(cornersroi.left_bottom.y, cornersroi.right_bottom.y);
	roi.x += startx;
	roi.y += starty;
	roi.width = endx - startx;
	roi.height = endy - starty;
}

void timeCounter(time_t start)
{
	time_t end = clock();
	double interval = double(end - start) / CLOCKS_PER_SEC;
	int i = 1;
	cout << "interval: " << interval << ";";
}

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	//左上角(0,0,1)
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

void CalcROICorners(const Mat& H, const Rect & roi)
{
	//左上角(roi.x, roi.y, 1)
	double v2[] = { roi.x, roi.y, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.left_top.x = v1[0] / v1[2];
	cornersroi.left_top.y = v1[1] / v1[2];

	//右上角(roi.x + roi.width, roi.y, 1)
	v2[0] = roi.x + roi.width;
	v2[1] = roi.y;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.right_top.x = v1[0] / v1[2];
	cornersroi.right_top.y = v1[1] / v1[2];

	//左下角(roi.x, roi.y + roi.height, 1)
	v2[0] = roi.x;
	v2[1] = roi.y + roi.height;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.left_bottom.x = v1[0] / v1[2];
	cornersroi.left_bottom.y = v1[1] / v1[2];

	//右:下角(roi.x, roi.y + roi.height, 1)
	v2[0] = roi.x + roi.width;
	v2[1] = roi.y + roi.height;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.right_bottom.x = v1[0] / v1[2];
	cornersroi.right_bottom.y = v1[1] / v1[2];
}

vector<Mat> getFiles(cv::String dir)
{
	glob(dir, paths, false);

	vector<Mat> images;
	for ( auto path : paths )
	{
		images.push_back(imread(path));
	}
	return images;
}

