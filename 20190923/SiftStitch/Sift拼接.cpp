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
Mat doStitchTwo(Mat & img1, Mat & img2);
vector<Mat> getFiles(cv::String dir);
void timeCounter(time_t start);
Rect roi(100, 800, 10, 10);

vector<cv::String> paths;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;
string dir = "images";

void stitch()
{
	cout << "stitch start!\n";
	time_t begin = clock();

	vector<Mat> images = getFiles(dir);

#if 1
	Mat img0 = images[0];
	Mat img1 = images[1];
	cout << "stitching \"" << paths[1] << "\" ";
	Mat dst = doStitchTwo(img0, img1);

	//Mat ttt = imread("res.png");
	Mat img2 = images[2];
	cout << "stitching \"" << paths[2] << "\" ";
	dst = doStitchTwo(dst, img2);

#else
	Mat img0 = imread("res_0_1.png");
	Mat img1 = imread("images/02.png");

	Mat dst = doStitchTwo(img0, img1);
#endif

#if 0
	int count = images.size();
	for (int i = 2; i < 4; i++)
	{
		cout << "stitching \"" << paths[i] << "\" ";
		dst = stitchTwo(dst, images[i]);
	}
#endif

	namedWindow("拼接效果", WINDOW_NORMAL);
	imshow("拼接效果", dst);
	imwrite("res.png", dst);

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "stitching end, total interval: " << interval << endl;

	//string name = "res_0_" +paths.size()+ ".png";
	//imwrite("res.png", dst);

	waitKey(0);
}

int main()
{
	stitch();

	return 0;
}

Mat findROI(Mat src)
{
	Mat res = src(roi); // 00+01
	//Mat res = src(Rect(100, 800, 2048, 2048)); // 00+01
	//namedWindow("roi", WINDOW_NORMAL);
	//imshow("roi", res);
	return res;
}

Mat stitchTwo(Mat & img1, Mat & img2)
{
	int width = img2.cols;
	int height = img2.rows;
	roi.width = width;
	roi.height = height;
	Mat img1roi = findROI(img1);

	Mat temp = doStitchTwo(img1roi, img2);

	Mat dst;
	int addwidth = 420;
	int addheight = 0.2 * height;
	copyMakeBorder(img1, dst, 0, addheight, addwidth, 0, 0, Scalar(0, 0, 0));

	int hx = temp.rows - height;
	int wx = temp.cols - width;
	//temp.copyTo(dst(Rect(700 - wx, 1200, temp.cols, temp.rows)));
	temp.copyTo(dst(Rect(roi.x, roi.y, temp.cols, temp.rows)));

	//update ROI
	roi.x = roi.x + addwidth + corners.left_bottom.x * 2;
	roi.y = roi.y + (corners.right_bottom.y - corners.left_top.y - height);

	//rectangle(dst, cvPoint(roi.x, roi.y), cvPoint(roi.x+roi.width, roi.y+roi.height), Scalar(0, 0, 255), 1, 1, 0);

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
	int addbottom = 0;
	int addleft = 420; //小于500出现拼接模糊
	int addright = 0;
	//copyMakeBorder(img2, imageMatch, addh, addbottom , addleft, addw, 0, Scalar(0, 0, 0));
	copyMakeBorder(img2, imageMatch, addtop, addbottom + addh, addleft+addw, addright, 0, Scalar(0, 0, 0));
	int h = imageMatch.rows * 0.2;
	copyMakeBorder(img1, imageSrc, addtop, addbottom + h, addleft, addright, 0, Scalar(0, 0, 0));

	Ptr<SIFT> sift; //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2, 否则即使配置好了还是显示SIFT为未声明的标识符  
	sift = SIFT::create(800);

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

	//timeCounter(begin);

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
#if 0
	int startx = min(corners.left_top.x, corners.left_bottom.x);
	int starty = min(corners.left_top.y, corners.right_top.y);
	int endx = max(corners.right_top.x, corners.right_bottom.x);
	int endy = min(corners.left_bottom.y, corners.right_bottom.y);
	roi.x = startx;
	roi.y = starty;
	roi.width = endx - startx;
	roi.height = endy - starty;

	namedWindow("imageMatch", WINDOW_NORMAL);
	imshow("imageMatch", imageMatch);
#endif

	//图像配准
	Mat imageWrap; // , imageTransform2;
	//warpPerspective(imageMatch, imageWrap, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), imageMatch.rows * 1.5)); //透视变换
	warpPerspective(imageMatch, imageWrap, homo, Size(imageMatch.cols, imageMatch.rows+h)); //透视变换
	//warpPerspective(imageSrc, imageTransform2, adjustMat*homo, Size(b.cols*1.3, b.rows*1.8));
	//namedWindow("旋转效果", WINDOW_NORMAL);
	//imshow("旋转效果", imageWrap);

//	rectangle(imageWrap, cvPoint(startx, starty), cvPoint(endx, endy), Scalar(0, 0, 255), 2, 2, 0);
//	namedWindow("imageWrap", WINDOW_NORMAL);
//	imshow("imageWrap", imageWrap);

	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageWrap.cols;
	int dst_height = imageWrap.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	//time_t mid = clock();
	//double interval1 = double(mid - begin) / CLOCKS_PER_SEC;
	//cout << "interval1: " << interval1 << " ";

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

	//timeCounter(begin);
	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "interval: " << interval << endl;

	//namedWindow("临时拼接效果", WINDOW_NORMAL);
	//imshow("临时拼接效果", dst);

	return dst;
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
	glob(dir, paths, false);

	vector<Mat> images;
	for ( auto path : paths )
	{
		images.push_back(imread(path));
	}
	return images;
}

