#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "path.h"

using namespace cv;
using namespace std;

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
	//const Mat dst = imread(MediaPath + "/DropX/reg/Sp0_P6_0_C4_lite_reg.png", 0);
	//const Mat src = imread(MediaPath + "/DropX/Sp0_P6_0_C4_lite.png", 0);
	Mat dst = imread(MediaPath + "/DropX/reg/Sp0_P6_0_C1_lite_reg.png");
	const Mat src = imread(MediaPath + "/DropX/Sp0_P6_0_C1_lite.png");
	//const Mat dst = imread(MediaPath + "/DropX/reg/Sp0_P6_0_C0_lite_reg.png", 0);
	//const Mat src = imread(MediaPath + "/DropX/Sp0_P6_0_C0_lite.png", 0);
	//const Mat dst = imread(MediaPath + "/DropX/reg/Sp0_P5_0_C4_lite_reg.png", 0);
	//const Mat src = imread(MediaPath + "/DropX/Sp0_P5_0_C4_lite.png", 0);
	//const Mat dst = imread(MediaPath + "/DropX/reg/Sp0_P5_0_C1_lite_reg.png", 0);
	//const Mat src = imread(MediaPath + "/DropX/Sp0_P5_0_C1_lite.png", 0);
	//const Mat dst = imread(MediaPath + "/DropX/reg/Sp0_P5_0_C0_lite_reg.png", 0);
	//const Mat src = imread(MediaPath + "/DropX/Sp0_P5_0_C0_lite.png", 0);

	namedWindow("原始图像", WINDOW_NORMAL);
	imshow("原始图像", src);

	namedWindow("第一次标定结果", WINDOW_NORMAL);
	imshow("第一次标定结果", dst);

	/*
	Mat gamma = work(src);
	namedWindow("GAMMA", WINDOW_NORMAL);
	imshow("GAMMA", gamma);

	Mat sub = src - gamma;
	namedWindow("SUB", WINDOW_NORMAL);
	imshow("SUB", sub);

	sub = gamma - src;
	namedWindow("SUB2", WINDOW_NORMAL);
	imshow("SUB2", sub);
	*/

	int row = src.rows;
	int col = src.cols;
	Mat sub(row, col, CV_16SC1);
	sub = src - dst;
	namedWindow("SUB3", CV_WINDOW_NORMAL);
	imshow("SUB3", sub);

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

#if 0
#include <cmath>
#include <iostream>  

#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>  
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "path.h"

using namespace cv;
using namespace std;

float get_Gamma_Value(Mat& gray_img);

void create_Gamma_Table(unsigned char* gama_table, float gama_value);

void Gamma_Correction(Mat& gray_img, Mat& dst_img, unsigned char* gama_table);

int main(int argc, char* argv[])
{
#if 0
	char input_image_name[100];
	char output_image_name[100];
	int image_num = 476;

	for (int i = 1; i <= image_num; i++)
	{
		sprintf(input_image_name, "../%s\\%d.jpg", "temp", i);
		Mat input_image = imread(input_image_name);

		if (input_image.empty())
		{
			cout << "Failed to load image !" << endl;
			continue;;
		}

		Mat gray_image;
		cvtColor(input_image, gray_image, CV_BGR2GRAY);

		// Start a timer
		double duration;
		duration = static_cast<double>(cv::getTickCount());

		float gama_value = get_Gamma_Value(gray_image);

		unsigned char LUT[256];
		create_Gamma_Table(LUT, gama_value);

		Mat result_image(gray_image.rows, gray_image.cols, gray_image.type());
		Gamma_Correction(gray_image, result_image, LUT);

		// Calculate the time cost and print
		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency();
		std::cout << duration * 1000 << " ms" << std::endl;

		imshow("Source_Image", input_image);
		imshow("Gamma_Correction", result_image);
		//imwrite("test6.bmp",result_image);

		waitKey(1);
	}
#endif

	//Mat input_image = imread(MediaPath+);
	const Mat input_image = imread(MediaPath + "lightCali.jpg");

	if (input_image.empty())
	{
		cout << "Failed to load image !" << endl;
		return -1;
	}

	Mat gray_image;
	cvtColor(input_image, gray_image, CV_BGR2GRAY);

	// Start a timer
	double duration;
	duration = static_cast<double>(cv::getTickCount());

	float gama_value = get_Gamma_Value(gray_image);

	unsigned char LUT[256];
	create_Gamma_Table(LUT, gama_value);

	Mat result_image(gray_image.rows, gray_image.cols, gray_image.type());
	Gamma_Correction(gray_image, result_image, LUT);

	// Calculate the time cost and print
	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	std::cout << duration * 1000 << " ms" << std::endl;

	imshow("Source_Image", input_image);
	imshow("Gamma_Correction", result_image);
	//imwrite("test6.bmp",result_image);

	waitKey(0);

	return 0;
}

/****************************************************
①当Gamma值比1大时，在输入值相同的情况下，输出值减小；
②当Gamma值为1时，输出值不变；
③当Gamma值比1小时，在输入值相同的情况下，输出值增加。
****************************************************/
//公式：gamma = log(y/range)/ log(x/range)，x是整幅图像像素的平均值，y是像素值最大范围的一半。

//先计算灰度图像的像素均值mean，将计算出来的均值带入 gammaVal = log(mean/255)/log(0.5) 这个公式中，就可以得到Gamma值了。
float get_Gamma_Value(Mat& gray_img)
{
	if (gray_img.empty())
	{
		return -1.0;
	}

	cv::Scalar meam_value = cv::mean(gray_img);

	float val = meam_value.val[0];
	//float gamma_val = (log10(val / 255.0)) / (log10(0.5));
	float gamma_val = (log10(0.5)) / (log10(val / 255.0));

	return gamma_val;
}

void create_Gamma_Table(unsigned char* gama_table, float gama_value)
{
	for (int i = 0; i < 256; i++)
	{
		float f = (i + 0.5f) / 255.0;
		f = (float)(pow(f, gama_value));
		gama_table[i] = saturate_cast<uchar>(f * 255.0f - 0.5f);
	}
}

void Gamma_Correction(Mat& gray_img, Mat& dst_img, unsigned char* gama_table)
{
	if (gray_img.channels() != dst_img.channels() || gray_img.cols != dst_img.cols || gray_img.rows != dst_img.rows)
	{
		return;
	}

	for (int i = 0; i < gray_img.rows; i++)
	{
		for (int j = 0; j < gray_img.cols; j++)
		{
			dst_img.at<uchar>(i, j) = gama_table[(int)(gray_img.at<uchar>(i, j))];
		}
	}
}
#endif
