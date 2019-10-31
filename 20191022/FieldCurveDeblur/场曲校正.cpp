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
#include <ctime>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

void CalcCorners(const Mat & H, const Mat & src);
void CalcROICorners(const Mat& H, const Rect & roi);
Mat doStitchTwo(Mat & img1, Mat & img2);
// vector<Mat> getFiles(cv::String dir);
void getFiles(cv::String dir);
void timeCounter(string str, time_t start);
inline void updateROI();
Mat Optimize(Mat& img);
void stitchAndClip(int index);
Mat FundamentalRansac(vector<KeyPoint>& current_keypoint1, vector<KeyPoint>& current_keypoint2, vector<DMatch>& current_matches);

Rect roi(0, 0, 0, 0);
int g_width;
int g_height;
vector<Mat> g_images;
vector<KeyPoint> Ransac_keypoint1, Ransac_keypoint2;
vector<DMatch> Ransac_matches;

#if 1
string dir = "blur";
#define BORDERWIDTH 500
#define BORDERHEIGHT 50
#else
string dir = "blur";
#define BORDERWIDTH 800
#define BORDERHEIGHT 300
#endif

string dir_deblur = "deblur/";
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

void deblur()
{
	cout << "deblur start!\n";
	time_t begin = clock();

	getFiles(dir);

	Mat img0 = g_images[0];
	Mat img1 = g_images[1];
	g_height = img0.rows;
	g_width = img0.cols;

	// cout << "debluring \"" << paths[0] << "\" ";
	// stitchAndClip(0);

	int count = g_images.size();
	for (int i = 0; i < count - 2; i++)
	{
		cout << "debluring \"" << paths[i] << "\" ";
		stitchAndClip(i);
	}

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "deblur end, total interval: " << interval << endl;

	waitKey(0);
}

void stitchAndClip(int index)
{
	time_t begin = clock();
	Mat img0 = g_images[index];
	//Mat img1 = g_images[index+2];
	Mat img1 = g_images[index+1];

	Rect rect0(0, g_height / 2, g_width / 2, g_height / 2);
	//Rect rect1(g_width / 4, g_height / 4, g_width / 2, g_height / 2);
	Rect rect1(g_width * 3 / 8 - 60, g_height * 5 / 12, g_width / 2, g_height / 2);
	// Rect rect0(0, g_height * 2 / 3, g_width / 3, g_height / 3);
	// Rect rect1(g_width / 3, g_height / 3, g_width / 3, g_height / 3);
	Mat img0roi = img0(rect0);
	Mat img1roi = img1(rect1);

	Mat dst = doStitchTwo(img0roi, img1roi);
	updateROI();
	
	string path = paths[index];
	int pos = path.find_last_of('\\');
	string name(path.substr(pos + 1));
	string filename = dir_deblur + name;
	Rect roi2(dst.cols - g_width/2, 0,g_width/2, g_height/2);
	// Rect roi2(dst.cols - g_width/3, 0,g_width/3, g_height/3);
	Mat dstroi = dst(roi2);
	dstroi.copyTo(img0(rect0));
	imwrite(filename, img0);

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "interval: " << interval << endl;
}

int main()
{
	deblur();

	return 0;
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
	// cout << "addw: " << addw << "addh: " << addh << endl;

	// make border
	int addtop = 0;
	int addbottom = BORDERHEIGHT;
	int addleft = BORDERWIDTH;
	int addright = 0;
	// copyMakeBorder(img2, imageMatch, addh, addbottom , addleft, addw, 0, Scalar(0, 0, 0));
	copyMakeBorder(img2, imageMatch, addtop, addbottom + addh, addleft+addw, addright, 0, Scalar(0, 0, 0));
	int h = imageMatch.rows * 0.2;
	copyMakeBorder(img1, imageSrc, addtop, addbottom + h, addleft, addright, 0, Scalar(0, 0, 0));

	int rows = imageMatch.rows;
	int cols = imageMatch.cols;

	// 转灰度图
	Mat graySrc, grayMatch;
	cvtColor(imageSrc, graySrc, CV_BGR2GRAY);
	cvtColor(imageMatch, grayMatch, CV_BGR2GRAY);

	Ptr<SIFT> sift = SIFT::create(15000); // 实例化特征检测器

	FlannBasedMatcher matcher; //实例化Flann匹配器
	Mat descriptor1, descriptor2;
	vector<KeyPoint> key1, key2;
	vector<DMatch> matches;

	sift->detectAndCompute(grayMatch, Mat(), key1, descriptor1);
	sift->detectAndCompute(graySrc, Mat(), key2, descriptor2);

	// Mat keySrc, keyMatch;
	// drawKeypoints(graySrc, key2, keySrc);//画出特征点
	// imwrite("keySrc.png", keySrc);
	// drawKeypoints(grayMatch, key1, keyMatch);//画出特征点
	// imwrite("keyMatch.png", keyMatch);

	matcher.match(descriptor2, descriptor1, matches); //匹配，数据来源是特征向量，结果存放在DMatch类型里面  

	// sort函数对数据进行升序排列
	sort(matches.begin(), matches.end());     //筛选匹配点，根据match里面特征对的距离从小到大排序
	vector<DMatch> good_matches;
	int ptsPairs = std::min(500, (int)(matches.size()));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]); //距离最小的700个压入新的DMatch
	}

	// 画特征匹配图
	// Mat outimg;
	// drawMatches(imageMatch, key1, imageSrc, key2, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	// namedWindow("特征匹配效果", WINDOW_NORMAL);
	// imshow("特征匹配效果", outimg);

	// 计算图像配准点
	vector<KeyPoint> keypoint1, keypoint2;

	for (int i = 0; i < good_matches.size(); i++)
	{
		keypoint1.push_back(key1[good_matches[i].trainIdx]);
		keypoint2.push_back(key2[good_matches[i].queryIdx]);
	}

	std::cout << "count of keypoints: " << good_matches.size() << std::endl;

	/*
	// Ransac消除误匹配
	int times = 0, current_num = 1, per_num = 0;;
	Mat img_Ransac_matches;
	char window_name[] = "0次Ransac之后匹配结果";
	Mat Fundamental;
	while (1)
	{
		if (per_num != current_num) {
			Ransac_keypoint1.clear();
			Ransac_keypoint2.clear();
			Ransac_matches.clear();
			per_num = good_matches.size();
			Fundamental = FundamentalRansac(keypoint1, keypoint2, good_matches);
			cout << endl << "Ransac" << ++times << "次之后的匹配点数为：" << Ransac_matches.size() << endl;
			cvWaitKey(1);
			keypoint1.clear();
			keypoint2.clear();
			good_matches.clear();
			keypoint1 = Ransac_keypoint1;
			keypoint2 = Ransac_keypoint2;
			good_matches = Ransac_matches;
			current_num = good_matches.size();
		}
		else
			break;
	}
	*/

	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i < good_matches.size(); i++)
	{
		imagePoints1.push_back(keypoint1[i].pt);
		imagePoints2.push_back(keypoint2[i].pt);
	}

	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);

	// 计算配准图的四个顶点坐标
	CalcCorners(homo, imageMatch);

	Rect roi = Rect(imageMatch.cols - g_width, 0, g_width, g_height);
	CalcROICorners(homo, roi);

	// 透视变换
	Mat imageWrap;
	warpPerspective(imageMatch, imageWrap, homo, Size(imageMatch.cols, imageMatch.rows+h));

	int dst_width = imageWrap.cols;
	int dst_height = imageWrap.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	for (int i = 0; i < dst_height; ++i) {
		for (int j = 0; j < dst_width; ++j) {
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
	// cout << "interval: " << interval << endl;

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

void timeCounter(string massege,time_t start)
{
	time_t end = clock();
	double interval = double(end - start) / CLOCKS_PER_SEC;
	int i = 1;
	cout << massege << ", interval: " << interval << "\n";
}

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	// 左上角(0,0,1)
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	// 左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	// 右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	// 右下角(src.cols,src.rows,1)
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
	// 左上角(roi.x, roi.y, 1)
	double v2[] = { double(roi.x), double(roi.y), 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.left_top.x = v1[0] / v1[2];
	cornersroi.left_top.y = v1[1] / v1[2];

	// 右上角(roi.x + roi.width, roi.y, 1)
	v2[0] = roi.x + roi.width;
	v2[1] = roi.y;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.right_top.x = v1[0] / v1[2];
	cornersroi.right_top.y = v1[1] / v1[2];

	// 左下角(roi.x, roi.y + roi.height, 1)
	v2[0] = roi.x;
	v2[1] = roi.y + roi.height;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.left_bottom.x = v1[0] / v1[2];
	cornersroi.left_bottom.y = v1[1] / v1[2];

	// 右:下角(roi.x, roi.y + roi.height, 1)
	v2[0] = roi.x + roi.width;
	v2[1] = roi.y + roi.height;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi.right_bottom.x = v1[0] / v1[2];
	cornersroi.right_bottom.y = v1[1] / v1[2];
}

void getFiles(cv::String dir)
{
	glob(dir, paths, false);

	// vector<Mat> images;
	for ( auto path : paths )
	{
#if 1
		g_images.push_back(imread(path));
#else
		Mat img = imread(path);
		Mat dst;
		resize(img, dst, Size(), 0.5, 0.5);
		g_images.push_back(dst);
#endif
	}
	// return images;
}

Mat Optimize(Mat& img)
{
	// time_t begin = clock();
	int rows = img.rows;
	int cols = img.cols;
	Mat gray = img;
	if (3 == img.channels())
	{
		cvtColor(img, gray, COLOR_BGR2GRAY);
	}

	int left = 0;
	int bottom = 0;

	// 下边界
	for (int i = rows-1; i >= 0; i--)
	{
		for (int j = cols - 1; j >= 0; j--)
		{
			if (gray.at<uchar>(i, j) != 0)
			{
				bottom = i;
				goto findLeft;
			}
		}
	}

findLeft:
	// 左边界
	for (int i = 0; i < cols; i++)
	{
		for (int j = rows - 1; j >= 0; j--)
		{
			if (gray.at<uchar>(j, i) != 0)
			{
				left = i;
				goto end;
			}
		}
	}

end:
	// cout << "left: " << left << ", bottom: " << bottom << endl;
	// timeCounter(begin);
	return img(Rect(left, 0, cols-left, bottom));
}

Mat FundamentalRansac(vector<KeyPoint>& current_keypoint1, vector<KeyPoint>& current_keypoint2, vector<DMatch>& current_matches)
{
	vector<Point2f>p1, p2;
	for (size_t i = 0; i < current_matches.size(); i++)
	{
		p1.push_back(current_keypoint1[i].pt);
		p2.push_back(current_keypoint2[i].pt);
	}

	vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(p1, p2, RansacStatus, FM_RANSAC);
	int index = 0;
	for (size_t i = 0; i < current_matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			Ransac_keypoint1.push_back(current_keypoint1[i]);
			Ransac_keypoint2.push_back(current_keypoint2[i]);
			current_matches[i].queryIdx = index;
			current_matches[i].trainIdx = index;
			Ransac_matches.push_back(current_matches[i]);
			index++;
		}
	}
	return Fundamental;
}


