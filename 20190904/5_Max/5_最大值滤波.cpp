#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "path.h"

using namespace cv;
using namespace std;
	
//返回最大值
uchar MaxValue(uchar pixel0, uchar pixel1, uchar pixel2, uchar pixel3,
	uchar pixel4, uchar pixel5, uchar pixel6, uchar pixel7, uchar pixel8)
{
	uchar array[9];
	array[0] = pixel0;
	array[1] = pixel1;
	array[2] = pixel2;
	array[3] = pixel3;
	array[4] = pixel4;
	array[5] = pixel5;
	array[6] = pixel6;
	array[7] = pixel7;
	array[8] = pixel8;
	uchar max = array[0];
	for (size_t i = 0; i < 9; i++)
	{
		if(array[i] > max)
			max = array[i];
	}
	return max;
}

void MaxValueFilter(const Mat &src, Mat &dst)
{
	if (!src.data)
		return;
	Mat _dst(src.size(), src.type());
	int row = src.rows;
	int col= src.cols;
	if (src.channels() == 3)
	{
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
					_dst.at<Vec3b>(i, j)[0] = MaxValue(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
						src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
						src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
						src.at<Vec3b>(i - 1, j - 1)[0]);
					_dst.at<Vec3b>(i, j)[1] = MaxValue(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
						src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
						src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
						src.at<Vec3b>(i - 1, j - 1)[1]);
					_dst.at<Vec3b>(i, j)[2] = MaxValue(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
						src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
						src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
						src.at<Vec3b>(i - 1, j - 1)[2]);
				}
				else
					_dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
			}
		}
	}
	else if (src.channels() == 1)
	{
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if ((i - 1) > 0 && (i + 1) < row && (j - 1) > 0 && (j + 1) < col) {
					_dst.at<uchar>(i, j) = MaxValue( src.at<uchar>(i-1, j-1), src.at<uchar>(i-1, j), src.at<uchar>(i-1, j+1), src.at<uchar>(i, j-1),
												src.at<uchar>(i, j), src.at<uchar>(i, j+1), src.at<uchar>(i+1, j-1), src.at<uchar>(i+1, j),
												src.at<uchar>(i+1, j+1) );
				} else {
					_dst.at<uchar>(i, j) = src.at<uchar>(i, j);
				}
			}
		}
	}
	_dst.copyTo(dst);//拷贝
	imshow("MaxValueFilter", dst);
}

int main()
{
	Mat src = imread(MediaPath + "test.bmp", 0);
	if (!src.data)
	{
		cout << "打开原图失败\n";
		return -1;
	}
	imshow("原图", src);
	cout << src.channels() << endl;

	Mat dst(src.size(), src.type());
	MaxValueFilter(src, dst);

	waitKey(0);

	return 0;
}
