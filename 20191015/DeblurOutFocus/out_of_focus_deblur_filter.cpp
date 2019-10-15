﻿/**
* @brief You will learn how to recover an out-of-focus image by Wiener filter
* @author Karpushin Vladislav, karpushin@ngs.ru, https://github.com/VladKarpushin
*/
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void help();
void calcPSF(Mat& outputImg, Size filterSize, int R);
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma = 5.0, double beta = 0.2);

const String keys =
"{help h usage ? |                 | print this message   }"
"{image          |4c6.png | input image name     }"
"{R              |53               | radius               }"
"{SNR            |5200             | signal to noise ratio}"
;

int main(int argc, char *argv[])
{
	// help();
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	int R = parser.get<int>("R");
	int snr = parser.get<int>("SNR");
	string strInFileName = parser.get<String>("image");

	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	Mat imgIn;
	imgIn = imread(strInFileName, IMREAD_GRAYSCALE);
	//normalize(imgIn, imgIn, 0, 255, NORM_MINMAX);
	//imwrite("in.png", imgIn);
	if (imgIn.empty()) //check whether the image is loaded or not
	{
		cout << "ERROR : Image cannot be loaded..!!" << endl;
		return -1;
	}

	Mat imgOut;

	//! [main]
	// it needs to process even image only
	Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);

	//Hw calculation (start)
	Mat Hw, h;
	calcPSF(h, roi.size(), R);
	calcWnrFilter(h, Hw, 1.0 / double(snr));
	//Hw calculation (stop)

	imgIn.convertTo(imgIn, CV_32F);
	edgetaper(imgIn, imgIn);

	// filtering (start)
	filter2DFreq(imgIn(roi), imgOut, Hw);
	// filtering (stop)
	//! [main]

	imgOut.convertTo(imgOut, CV_8U);
	normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
	imwrite("result.jpg", imgOut);
	//normalize(h, h, 0, 255, NORM_MINMAX);
	//h.convertTo(h, CV_8U);
	//imwrite("psf.png", h);
	return 0;
}

void help()
{
	cout << "2018-07-12" << endl;
	cout << "DeBlur_v8" << endl;
	cout << "You will learn how to recover an out-of-focus image by Wiener filter" << endl;
}

//! [calcPSF]
void calcPSF(Mat& outputImg, Size filterSize, int R)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	circle(h, point, R, 255, -1, 8);
	Scalar summa = sum(h);
	outputImg = h / summa[0];
}
//! [calcPSF]

//! [fftshift]
void fftshift(const Mat& inputImg, Mat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));
	Mat q1(outputImg, Rect(cx, 0, cx, cy));
	Mat q2(outputImg, Rect(0, cy, cx, cy));
	Mat q3(outputImg, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}
//! [fftshift]

//! [filter2DFreq]
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
	Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_SCALE);

	Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
	Mat complexH;
	merge(planesH, 2, complexH);
	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0);

	idft(complexIH, complexIH);
	split(complexIH, planes);
	outputImg = planes[0];
}
//! [filter2DFreq]

//! [calcWnrFilter]
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
	Mat h_PSF_shifted;
	fftshift(input_h_PSF, h_PSF_shifted);
	Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	Mat denom;
	pow(abs(planes[0]), 2, denom);
	denom += nsr;
	divide(planes[0], denom, output_G);
}
//! [calcWnrFilter]

//! [edgetaper]
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
{
	int Nx = inputImg.cols;
	int Ny = inputImg.rows;
	Mat w1(1, Nx, CV_32F, Scalar(0));
	Mat w2(Ny, 1, CV_32F, Scalar(0));

	float* p1 = w1.ptr<float>(0);
	float* p2 = w2.ptr<float>(0);
	float dx = float(2.0 * CV_PI / Nx);
	float x = float(-CV_PI);
	for (int i = 0; i < Nx; i++)
	{
		p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
		x += dx;
	}
	float dy = float(2.0 * CV_PI / Ny);
	float y = float(-CV_PI);
	for (int i = 0; i < Ny; i++)
	{
		p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
		y += dy;
	}
	Mat w = w2 * w1;
	multiply(inputImg, w, outputImg);
}
//! [edgetaper]
