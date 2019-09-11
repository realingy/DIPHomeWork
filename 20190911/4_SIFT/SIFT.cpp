/*
#include "highgui/highgui.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  
#include "opencv2/legacy/legacy.hpp" 
#include <vector>

using namespace cv;
using namespace std;

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
Point2f getTransformPoint(const Point2f originalPoint, const Mat &transformMaxtri);

int main(int argc, char *argv[])
{
	Mat image01 = imread(argv[1]);
	Mat image02 = imread(argv[2]);
	imshow("拼接图像1", image01);
	imshow("拼接图像2", image02);

	//灰度图转换
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);

	//提取特征点  
	SiftFeatureDetector siftDetector(800);  // 海塞矩阵阈值
	vector<KeyPoint> keyPoint1, keyPoint2;
	siftDetector.detect(image1, keyPoint1);
	siftDetector.detect(image2, keyPoint2);

	//特征点描述，为下边的特征点匹配做准备  
	SiftDescriptorExtractor siftDescriptor;
	Mat imageDesc1, imageDesc2;
	siftDescriptor.compute(image1, keyPoint1, imageDesc1);
	siftDescriptor.compute(image2, keyPoint2, imageDesc2);

	//获得匹配特征点，并提取最优配对  	
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	sort(matchePoints.begin(), matchePoints.end()); //特征点排序	
	//获取排在前N个的最优匹配特征点
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i < 10; i++)
	{
		imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);
		imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵，尺寸为3*3
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, image01.cols, 0, 1.0, 0, 0, 0, 1.0);
	Mat adjustHomo = adjustMat * homo;

	//获取最强配对点在原始图像和矩阵变换后图像上的对应位置，用于图像拼接点的定位
	Point2f originalLinkPoint, targetLinkPoint, basedImagePoint;
	originalLinkPoint = keyPoint1[matchePoints[0].queryIdx].pt;
	targetLinkPoint = getTransformPoint(originalLinkPoint, adjustHomo);
	basedImagePoint = keyPoint2[matchePoints[0].trainIdx].pt;

	//图像配准
	Mat imageTransform1;
	warpPerspective(image01, imageTransform1, adjustMat*homo, Size(image02.cols + image01.cols + 10, image02.rows));

	//在最强匹配点的位置处衔接，最强匹配点左侧是图1，右侧是图2，这样直接替换图像衔接不好，光线有突变
	Mat ROIMat = image02(Rect(Point(basedImagePoint.x, 0), Point(image02.cols, image02.rows)));
	ROIMat.copyTo(Mat(imageTransform1, Rect(targetLinkPoint.x, 0, image02.cols - basedImagePoint.x + 1, image02.rows)));

	namedWindow("拼接结果", 0);
	imshow("拼接结果", imageTransform1);
	waitKey();
	return 0;
}

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
Point2f getTransformPoint(const Point2f originalPoint, const Mat &transformMaxtri)
{
	Mat originelP, targetP;
	originelP = (Mat_<double>(3, 1) << originalPoint.x, originalPoint.y, 1.0);
	targetP = transformMaxtri * originelP;
	float x = targetP.at<double>(0, 0) / targetP.at<double>(2, 0);
	float y = targetP.at<double>(1, 0) / targetP.at<double>(2, 0);
	return Point2f(x, y);
}
*/


/*
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include"opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

int main()
{
    Mat a = imread("21.jpg");    //读取灰度图像
    Mat b = imread("22.jpg");

    Ptr<SURF> surf;      //创建方式和2中的不一样
    surf = SURF::create(800);

    BFMatcher matcher;
    Mat c, d;
    vector<KeyPoint>key1, key2;
    vector<DMatch> matches;

    surf->detectAndCompute(a, Mat(), key1, c);
    surf->detectAndCompute(b, Mat(), key2, d);

    matcher.match(c, d, matches);       //匹配

    sort(matches.begin(), matches.end());  //筛选匹配点
    vector< DMatch > good_matches;
    int ptsPairs = std::min(50, (int)(matches.size() * 0.15));
    cout << ptsPairs << endl;
    for (int i = 0; i < ptsPairs; i++)
    {
        good_matches.push_back(matches[i]);
    }

    Mat outimg;
    drawMatches(a, key1, b, key2, good_matches, outimg, Scalar::all(-1), Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点
    imshow("out", outimg);
    waitKey(0);

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        obj.push_back(key1[good_matches[i].queryIdx].pt);
        scene.push_back(key2[good_matches[i].trainIdx].pt);
    }

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(a.cols, 0);
    obj_corners[2] = Point(a.cols, a.rows);
    obj_corners[3] = Point(0, a.rows);
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography(obj, scene, RANSAC);      //寻找匹配的图像
    perspectiveTransform(obj_corners, scene_corners, H);

    line(outimg,scene_corners[0] + Point2f((float)a.cols, 0), scene_corners[1] + Point2f((float)a.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);       //绘制
    line(outimg,scene_corners[1] + Point2f((float)a.cols, 0), scene_corners[2] + Point2f((float)a.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);
    line(outimg,scene_corners[2] + Point2f((float)a.cols, 0), scene_corners[3] + Point2f((float)a.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);
    line(outimg,scene_corners[3] + Point2f((float)a.cols, 0), scene_corners[0] + Point2f((float)a.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);
    imshow("aaaa",outimg);
    waitKey(0);
}
*/

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/core.hpp>
#include "path.h"

#include<iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界

	double processWidth = img1.cols - start;//重叠区域的宽度
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i); // 获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
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

int main()
{
	Mat srcImage1 = imread( MediaPath + "SIFT/1.jpg", 1);
	Mat srcImage2 = imread( MediaPath + "SIFT/2.jpg", 1);
	if (!srcImage1.data || !srcImage2.data)
	{
		cout << "读取图片出错" << endl;
		return false;
	}

	imshow("原始图1", srcImage1);
	imshow("原始图2", srcImage2);

	int minHessian = 100;
	Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(minHessian);

	vector<cv::KeyPoint> key_points_1, key_points_2;

	Mat dstImage1, dstImage2;
	detector->detectAndCompute(srcImage1, Mat(), key_points_1, dstImage1);
	detector->detectAndCompute(srcImage2, Mat(), key_points_2, dstImage2);//可以分成detect和compute

	Mat img_keypoints_1, img_keypoints_2;
	// drawKeypoints(srcImage1,key_points_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
	// drawKeypoints(srcImage2, key_points_2, img_keypoints_2, Scalar::all(-1),DrawMatchesFlags::DEFAULT);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	vector<DMatch>mach;

	matcher->match(dstImage1, dstImage2, mach);
	double Max_dist = 0;
	double Min_dist = 100;
	for (int i = 0; i < dstImage1.rows; i++)
	{
		double dist = mach[i].distance;
		if (dist < Min_dist)Min_dist = dist;
		if (dist > Max_dist)Max_dist = dist;
	}
	cout << "最短距离" << Min_dist << endl;
	cout << "最长距离" << Max_dist << endl;

	vector<DMatch>goodmaches;
	for (int i = 0; i < dstImage1.rows; i++)
	{
		if (mach[i].distance < 2 * Min_dist)
			goodmaches.push_back(mach[i]);
	}
	Mat img_maches;
	drawMatches(srcImage1, key_points_1, srcImage2, key_points_2, goodmaches, img_maches);

	for (int i = 0; i < goodmaches.size(); i++)
	{
		cout << "符合条件的匹配：" << goodmaches[i].queryIdx << "--" << goodmaches[i].trainIdx << endl;
	}
	// imshow("效果图1", img_keypoints_1);
	 //imshow("效果图2", img_keypoints_2);
	imshow("匹配效果", img_maches);
	waitKey(0);
	Mat mat1;
	OptimizeSeam(srcImage1, img_maches, mat1);
	imshow("匹配效果", mat1);
	waitKey(0);
	return 0;
}
