#include <opencv2/core.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <ctime>
#include <iostream>

using namespace cv::xfeatures2d;
using namespace cv::ml;

using namespace cv;
using namespace std;

void timeCounter(string str, time_t start);

void fast(const Mat & img)
{
	time_t begin = clock();

	std::vector<KeyPoint> keys;
	FAST(img, keys, 30);

	timeCounter("fast1: ", begin);

	/*
	Ptr<SIFT> surf;
	surf = SIFT::create();

	Mat keyMat;
	surf->compute(img, keys, keyMat);
	*/

	// Draw keypoints 
	//drawKeypoints(img, keys, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img, keys, img);//画出特征点

	// Show detected (drawn) keypoints 
	namedWindow("fast", WINDOW_NORMAL);
	imshow("fast", img);

	timeCounter("fast: ", begin);
}

void fast2(const Mat & img)
{
	time_t begin = clock();

	Ptr<FastFeatureDetector> fast;
	fast = FastFeatureDetector::create(7000);

	Mat keyMat;
	vector<KeyPoint> keys;

	fast->detect(img, keys);

	//fast->compute(img, keys, keyMat);
	//Ptr<BriefDescriptorExtractor> descriptor;
	//descriptor = BriefDescriptorExtractor::create(7000);
	
	//Ptr<DAISY> descriptor;
	//descriptor = DAISY::create();

	//descriptor->compute(img, keys, keyMat);

	drawKeypoints(img, keys, img);//画出特征点
	// Show detected (drawn) keypoints 
	namedWindow("fast2", WINDOW_NORMAL);
	imshow("fast2", img);
	//namedWindow("fast2keymat", WINDOW_NORMAL);
	//imshow("fast2keymat", keyMat);

	timeCounter("fast2: ", begin);
}

void sift(const Mat & img)
{
	time_t begin = clock();

	Ptr<SIFT> sift; //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2, 否则即使配置好了还是显示SIFT为未声明的标识符  
	sift = SIFT::create(10000);

	Mat keyMat;
	vector<KeyPoint> keys;

	//sift->detect(img, keys, Mat());
	//timeCounter("sift1: ", begin);
	//sift->compute(img, keys, keyMat);
	sift->detectAndCompute(img, Mat(), keys, keyMat);

	drawKeypoints(img, keys, img);//画出特征点
	// Show detected (drawn) keypoints 
	namedWindow("sift", WINDOW_NORMAL);
	imshow("sift", img);

	timeCounter("sift: ", begin);
}

void surf(const Mat & img)
{
	time_t begin = clock();

	Ptr<SURF> sift; //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2, 否则即使配置好了还是显示SIFT为未声明的标识符  
	sift = SURF::create(7000);

	Mat keyMat;
	vector<KeyPoint> keys;

	sift->detect(img, keys, Mat());

	timeCounter("surf1: ", begin);
	sift->compute(img, keys, keyMat);
	//sift->detectAndCompute(img, Mat(), keys, keyMat);

	drawKeypoints(img, keys, img);//画出特征点
	// Show detected (drawn) keypoints 
	namedWindow("surf", WINDOW_NORMAL);
	imshow("surf", img);

	timeCounter("surf2: ", begin);
}

int main(int argc, char** argv)
{
	Mat img = imread("02.png");
	//Mat img = imread("1.jpg");

	//fast(img);
	//fast2(img);
	sift(img);
	//surf(img);

	waitKey(0);
	return 0;
}

void timeCounter(string str, time_t start)
{
	time_t end = clock();
	double interval = double(end - start) / CLOCKS_PER_SEC;
	int i = 1;
	cout << str << "interval: " << interval << "\n";
}
