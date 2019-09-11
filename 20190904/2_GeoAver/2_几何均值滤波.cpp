#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "path.h"

using namespace cv;
using namespace std;

//几何均值滤波
void GeoAverFliter(const Mat &src, Mat &dst)
{
	Mat _dst(src.size(), CV_32FC1);
	double power = 1.0 / 9;
	cout << "power:" << power << endl;
	double geo = 1;
	if (src.channels() == 1) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
					if (src.at<uchar>(i, j) != 0) geo = geo * src.at<uchar>(i, j);
					if (src.at<uchar>(i + 1, j + 1) != 0) geo = geo * src.at<uchar>(i + 1, j + 1);
					if (src.at<uchar>(i + 1, j) != 0) geo = geo * src.at<uchar>(i + 1, j);
					if (src.at<uchar>(i, j + 1) != 0) geo = geo * src.at<uchar>(i, j + 1);
					if (src.at<uchar>(i + 1, j - 1) != 0) geo = geo * src.at<uchar>(i + 1, j - 1);
					if (src.at<uchar>(i - 1, j + 1) != 0) geo = geo * src.at<uchar>(i - 1, j + 1);
					if (src.at<uchar>(i - 1, j) != 0) geo = geo * src.at<uchar>(i - 1, j);
					if (src.at<uchar>(i, j - 1) != 0) geo = geo * src.at<uchar>(i, j - 1);
					if (src.at<uchar>(i - 1, j - 1) != 0) geo = geo * src.at<uchar>(i - 1, j - 1);
					_dst.at<float>(i, j) = pow(geo, power);
					geo = 1;
				}
				else
					_dst.at<float>(i, j) = src.at<uchar>(i, j);
			}
		}
	}
	_dst.convertTo(dst, CV_8UC1);

	imshow("geoAverFilter", dst);
}

int main()
{
	Mat src = imread(MediaPath + "test.bmp", 0);
	imshow("原图", src);
	cout << src.channels() << endl;

	Mat dst = Mat(src.size(), src.type());
	GeoAverFliter(src, dst);

	waitKey(0);

	return 0;
}
