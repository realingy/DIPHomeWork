#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

std::string MediaPath= "D:\\dir_git\\DIPHomeWork\\20190909\\data";

Mat RGB2HSV(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_32FC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float b = src.at<Vec3b>(i, j)[0] / 255.0;
			float g = src.at<Vec3b>(i, j)[1] / 255.0;
			float r = src.at<Vec3b>(i, j)[2] / 255.0;
			float minn = min(r, min(g, b));
			float maxx = max(r, max(g, b));
			dst.at<Vec3f>(i, j)[2] = maxx; //V
			float delta = maxx - minn;
			float h, s;
			if (maxx != 0) {
				s = delta / maxx;
			}
			else {
				s = 0;
			}
			if (r == maxx) {
				h = (g - b) / delta;
			}
			else if (g == maxx) {
				h = 2 + (b - r) / delta;
			}
			else {
				h = 4 + (r - g) / delta;
			}
			h *= 60;
			if (h < 0)
				h += 360;
			dst.at<Vec3f>(i, j)[0] = h;
			dst.at<Vec3f>(i, j)[1] = s;
		}
	}
	return dst;
}

Mat HSV2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	float r, g, b, h, s, v;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			h = src.at<Vec3f>(i, j)[0];
			s = src.at<Vec3f>(i, j)[1];
			v = src.at<Vec3f>(i, j)[2];
			if (s == 0) {
				r = g = b = v;
			}
			else {
				h /= 60;
				int offset = floor(h);
				float f = h - offset;
				float p = v * (1 - s);
				float q = v * (1 - s * f);
				float t = v * (1 - s * (1 - f));
				switch (offset)
				{
				case 0: r = v; g = t; b = p; break;
				case 1: r = q; g = v; b = p; break;
				case 2: r = p; g = v; b = t; break;
				case 3: r = p; g = q; b = v; break;
				case 4: r = t; g = p; b = v; break;
				case 5: r = v; g = p; b = q; break;
				default:
					break;
				}
			}
			dst.at<Vec3b>(i, j)[0] = int(b * 255);
			dst.at<Vec3b>(i, j)[1] = int(g * 255);
			dst.at<Vec3b>(i, j)[2] = int(r * 255);
		}
	}
	return dst;
}

Mat work(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat now = RGB2HSV(src);
	Mat H(row, col, CV_32FC1);
	Mat S(row, col, CV_32FC1);
	Mat V(row, col, CV_32FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			H.at<float>(i, j) = now.at<Vec3f>(i, j)[0];
			S.at<float>(i, j) = now.at<Vec3f>(i, j)[1];
			V.at<float>(i, j) = now.at<Vec3f>(i, j)[2];
		}
	}

	//高斯模糊的卷积核的尺寸必须是偶数
	int kernel_size = min(row, col);
	if (kernel_size % 2 == 0) {
		kernel_size -= 1;
	}

	float SIGMA1 = 15;
	float SIGMA2 = 80;
	float SIGMA3 = 250;
	float q = sqrt(2.0);
	Mat F(row, col, CV_32FC1);
	Mat F1, F2, F3;
	GaussianBlur(V, F1, Size(kernel_size, kernel_size), SIGMA1 / q);
	GaussianBlur(V, F2, Size(kernel_size, kernel_size), SIGMA2 / q);
	GaussianBlur(V, F3, Size(kernel_size, kernel_size), SIGMA3 / q);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			F.at <float>(i, j) = (F1.at<float>(i, j) + F2.at<float>(i, j) + F3.at<float>(i, j)) / 3.0;
		}
	}

	float average = mean(F)[0];
	Mat out(row, col, CV_32FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float gamma = powf(0.5, (average - F.at<float>(i, j)) / average);
			out.at<float>(i, j) = powf(V.at<float>(i, j), gamma);
		}
	}
	vector <Mat> v;
	v.push_back(H);
	v.push_back(S);
	v.push_back(out);
	Mat merge_;
	merge(v, merge_);
	Mat dst = HSV2RGB(merge_);
	return dst;
}

int main()
{
    const Mat src = imread(MediaPath + "/src/Sp2_P0_0_C1_lite.png");
    Mat cali = imread(MediaPath + "/cali/Sp2_P0_0_C1_lite.png");

	namedWindow("原始图像", WINDOW_NORMAL);
	imshow("原始图像", src);

	namedWindow("第一次标定结果", WINDOW_NORMAL);
	imshow("第一次标定结果", cali);

	Mat sub(src.size(), src.type());
	sub = src - cali;
	namedWindow("SUB", CV_WINDOW_NORMAL);
	imshow("SUB", sub);

	Mat gamma = work(src);
	namedWindow("GAMMA", WINDOW_NORMAL);
	imshow("GAMMA", gamma);

	/*
	Mat sub = src - gamma;
	namedWindow("SUB", WINDOW_NORMAL);
	imshow("SUB", sub);

	sub = gamma - src;
	namedWindow("SUB2", WINDOW_NORMAL);
	imshow("SUB2", sub);
	*/

	/*
	float average = mean(sub)[0];
	cout << "平均灰度值: " << average << endl;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			char p = sub.at<char>(i, j);
			if (p < 0) {
				sub.at<uchar>(i, j) -= 10;
			}  else {
				sub.at<uchar>(i, j) += 10;
			}
		}
	}

	Mat cali = src - sub;
	namedWindow("第二次校正", CV_WINDOW_NORMAL);
	imshow("第二次校正", cali);

	Mat sub3 = subb - sub;
	namedWindow("SUB5", CV_WINDOW_NORMAL);
	imshow("SUB5", sub3);
	average = mean(sub3)[0];
	cout << "平均灰度值: " << average << endl;
	*/

	/*
	average = mean(sub)[0];
	cout << "平均灰度值: " << average << endl;

//	namedWindow("SUB4", CV_WINDOW_NORMAL);
//	imshow("SUB4", sub);

	namedWindow("第二次校正", CV_WINDOW_NORMAL);
	imshow("第二次校正", cali);
	*/


	/*
	Mat sub = src - dst;
	namedWindow("SUB", WINDOW_NORMAL);
	imshow("SUB", sub);
	*/

	//imwrite(MediaPath + "/DropX/cali/Sp0_P6_0_C4_lite_cali.png", gamma);



#if 0
	//const Mat src_img = imread(MediaPath + "lightCali.jpg");
    const Mat src_img = imread(MediaPath + "dropx.png");

	if (src_img.empty())
	{
		printf("could not load image...\n");
		return -1;
	}

	//namedWindow("原图：", CV_WINDOW_AUTOSIZE);
	namedWindow("原图：", CV_WINDOW_NORMAL);
	imshow("原图：", src_img);

	Mat dst_img = work(src_img);
	//namedWindow("校正图：", CV_WINDOW_AUTOSIZE);
	namedWindow("校正图：", CV_WINDOW_NORMAL);
	imshow("校正图：", dst_img);
	//imwrite(MediaPath + "LightCaliEffec.jpg", dst_img);

	Mat sub = dst_img - src_img;
	namedWindow("灰度差：", CV_WINDOW_NORMAL);
	imshow("灰度差：", sub);
#endif

	waitKey(0);
	destroyAllWindows();

	return 0;

}



