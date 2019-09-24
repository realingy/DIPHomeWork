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
Mat stitchTwo(Mat & img1, Mat & img2);
vector<Mat> getFiles(cv::String dir);

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
	cout << "stitch start!\n";

	//Mat img1 = imread("images/00.png");
	//Mat img2 = imread("images/01.png");
	Mat img1 = imread("dst.png");
	Mat img2 = imread("images/02.png");

	Mat dst = stitchTwo(img1, img2);

	namedWindow("拼接效果", WINDOW_NORMAL);
	imshow("拼接效果", dst);
	//imwrite("dst.png", dst);

	cout << "stitch end!\n";

	waitKey(0);
	return 0;
}

Mat stitchTwo(Mat & img1, Mat & img2)
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

	// make border
	//int addtop = 0;
	int addbottom = 100;
	int addleft = 500;
	//int addright = 0;
	//cout << "addw: " << addw << "addh: " << addh << endl;
	copyMakeBorder(img2, imageMatch, 0, addbottom + addh, addleft, addw, 0, Scalar(0, 0, 0));
	int h = imageMatch.rows * 0.2;
	copyMakeBorder(img1, imageSrc, 0, addbottom + h, addleft, 0, 0, Scalar(0, 0, 0));

	Ptr<SIFT> sift; //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2, 否则即使配置好了还是显示SIFT为未声明的标识符  
	sift = SIFT::create(800);

	BFMatcher matcher; //实例化一个暴力匹配器
	Mat key_left, key_right;
	vector<KeyPoint> key1, key2;
	vector<DMatch> matches;    //DMatch是用来描述匹配好的一对特征点的类，包含这两个点之间的相关信息
							   //比如左图有个特征m，它和右图的特征点n最匹配，这个DMatch就记录它俩最匹配，并且还记录m和n的
							   //特征向量的距离和其他信息，这个距离在后面用来做筛选

	sift->detectAndCompute(imageMatch, Mat(), key1, key_left); //输入图像，输入掩码，输入特征点，输出Mat，存放所有特征点的描述向量
	sift->detectAndCompute(imageSrc, Mat(), key2, key_right); //这个Mat行数为特征点的个数，列数为每个特征向量的尺寸，SURF是64（维）

	matcher.match(key_right, key_left, matches);             //匹配，数据来源是特征向量，结果存放在DMatch类型里面  

	//sort函数对数据进行升序排列
	sort(matches.begin(), matches.end());     //筛选匹配点，根据match里面特征对的距离从小到大排序
	vector<DMatch> good_matches;
	int ptsPairs = std::min(100, (int)(matches.size() * 0.15));
	// cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);//距离最小的100个压入新的DMatch
	}

	//Mat outimg; //drawMatches这个函数直接画出摆在一起的图
	//drawMatches(imageMatch, key1, imageSrc, key1, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
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
	//cout << "left_top:" << corners.left_top << endl;
	//cout << "left_bottom:" << corners.left_bottom << endl;
	//cout << "right_top:" << corners.right_top << endl;
	//cout << "right_bottom:" << corners.right_bottom << endl;

	//图像配准
	Mat imageWrap; // , imageTransform2;
	//warpPerspective(imageMatch, imageWrap, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), imageMatch.rows * 1.5)); //透视变换
	warpPerspective(imageMatch, imageWrap, homo, Size(imageMatch.cols, imageMatch.rows+h)); //透视变换
	//warpPerspective(imageSrc, imageTransform2, adjustMat*homo, Size(b.cols*1.3, b.rows*1.8));
	//namedWindow("旋转效果", WINDOW_NORMAL);
	//imshow("旋转效果", imageWrap);

	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageWrap.cols;
	int dst_height = imageWrap.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	for (int i = 1; i < dst_height; ++i) {
		for (int j = 1; j < dst_width; ++j) {
			if(imageSrc.at<Vec3b>(i, j)[0] != 0 && imageWrap.at<Vec3b>(i, j)[0] == 0)
				dst.at<Vec3b>(i, j) = imageSrc.at<Vec3b>(i, j);
			else if(imageSrc.at<Vec3b>(i, j)[0] == 0 && imageWrap.at<Vec3b>(i, j)[0] != 0)
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j);
			else
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j) * 0.6 + imageSrc.at<Vec3b>(i, j) * 0.4;
		}
	}

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "interval: " << interval << endl;

	return dst;
}

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	//左上角(0,0,1)
	//cout << "V2: " << V2 << endl;
	//cout << "V1: " << V1 << endl;
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

vector<Mat> getFiles(cv::String dir)
{
	vector<cv::String> paths;
	glob(dir, paths, false);

	vector<Mat> images;
	//size_t count = fn.size(); //number of png files in images folder
	//for (size_t i = 0; i < count; i++)
	Mat img;
	for ( auto path : paths )
	{
		img = imread(path);
		images.push_back(img);
		//imshow("img", imread(fn[i]));
		//waitKey(1000);
	}
	return images;
}

