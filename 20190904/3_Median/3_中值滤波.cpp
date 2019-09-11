#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "path.h"

using namespace cv;
using namespace std;

//返回中值
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9)
{
	uchar arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//返回中值
}

//中值滤波
void MedianFliter(const Mat &src, Mat &dst) {
	if (!src.data) return;
	Mat _dst(src.size(), src.type());
	if (src.channels() == 3)
	{
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
					_dst.at<Vec3b>(i, j)[0] = Median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
						src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
						src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
						src.at<Vec3b>(i - 1, j - 1)[0]);
					_dst.at<Vec3b>(i, j)[1] = Median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
						src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
						src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
						src.at<Vec3b>(i - 1, j - 1)[1]);
					_dst.at<Vec3b>(i, j)[2] = Median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
						src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
						src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
						src.at<Vec3b>(i - 1, j - 1)[2]);
				}
				else
					_dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
			}
		}
	}
	if (src.channels() == 1)
	{
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
					_dst.at<uchar>(i, j) = Median(src.at<uchar>(i, j), src.at<uchar>(i + 1, j + 1),
						src.at<uchar>(i + 1, j), src.at<uchar>(i, j + 1), src.at<uchar>(i + 1, j - 1),
						src.at<uchar>(i - 1, j + 1), src.at<uchar>(i - 1, j), src.at<uchar>(i, j - 1),
						src.at<uchar>(i - 1, j - 1));

				}
				else
					_dst.at<uchar>(i, j) = src.at<uchar>(i, j);
			}
		}
	}
	_dst.copyTo(dst);//拷贝
	imshow("mediaFilter", dst);
}

int main()
{
	Mat src = imread(MediaPath + "test.bmp");
	imshow("原图", src);
	cout << src.channels() << endl;

	Mat dst = Mat(src.size(), src.type());
	MedianFliter(src, dst);

	waitKey(0);

	return 0;
}
