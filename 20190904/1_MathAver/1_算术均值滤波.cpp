#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "path.h"

using namespace cv;
using namespace std;


//算数均值滤波
void MathMediaFilter(Mat &src, Mat &dst) {
	if (src.channels() == 3)//彩色图像
	{
		for (int i = 1; i < src.rows; ++i) {
			for (int j = 1; j < src.cols; ++j) {
				if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) {
					//边缘不进行处理
					dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
						src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
						src.at<Vec3b>(i + 1, j)[0]) / 9;
					dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
						src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
						src.at<Vec3b>(i + 1, j)[1]) / 9;
					dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
						src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
						src.at<Vec3b>(i + 1, j)[2]) / 9;
				}
				else {//边缘赋值
					dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
					dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
					dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
				}
			}
		}
	}
	if (src.channels() == 1) {//灰度图像
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) {//边缘不进行处理
					dst.at<uchar>(i, j) = (src.at<uchar>(i, j) + src.at<uchar>(i - 1, j - 1) + src.at<uchar>(i - 1, j) + src.at<uchar>(i, j - 1) +
						src.at<uchar>(i - 1, j + 1) + src.at<uchar>(i + 1, j - 1) + src.at<uchar>(i + 1, j + 1) + src.at<uchar>(i, j + 1) +
						src.at<uchar>(i + 1, j)) / 9;
				}
				else {//边缘赋值
					dst.at<uchar>(i, j) = src.at<uchar>(i, j);
				}
			}
		}
	}

	imshow("MathMedianFilter", dst);

}

int main()
{
	Mat src = imread(MediaPath + "test.bmp");
	imshow("原图", src);
	cout << src.channels() << endl;

	Mat dst = src.clone();
	MathMediaFilter(src, dst);

	waitKey(0);

	return 0;
}
