#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

std::string MediaPath= "D:\\dir_git\\DIPHomeWork\\20190909\\data";

Mat eaualizeHist_GO(Mat src);
Mat aheGO(Mat src, int _step = 8);
Mat clheGO(Mat src, int _step = 8);
Mat claheGoWithoutInterpolation(Mat src, int _step = 8);
Mat claheGO(Mat src, int _step = 4);
Mat RGB2HSV(Mat src);

int main(int argc, char * argv[])
{
    //读入灰度图

    const Mat src = imread(MediaPath + "/src/Sp2_P0_0_C1_lite.png", 0);
    const Mat cali = imread(MediaPath + "/cali/Sp2_P0_0_C1_lite.png", 0);

    cv::namedWindow("原始图像", WINDOW_NORMAL);
    cv::imshow("原始图像",src);

	cv::namedWindow("第一次标定结果", WINDOW_NORMAL);
	cv::imshow("第一次标定结果", cali);

	Mat sub = src - cali;
    cv::namedWindow("SUB0", WINDOW_NORMAL);
    cv::imshow("SUB0", sub);

	sub = cali - src;
    cv::namedWindow("SUB1", WINDOW_NORMAL);
    cv::imshow("SUB1", sub);

    Mat clahe;
	clahe = claheGO(src);

    cv::namedWindow("CLAHE", WINDOW_NORMAL);
    cv::imshow("CLAHE", clahe);

	sub = src - clahe;
    cv::namedWindow("SUB3", WINDOW_NORMAL);
    cv::imshow("SUB3", sub);

//	imwrite(MediaPath + "/DropX/cali/Sp0_P6_0_C4_lite_cali.png", clahe);

	/*
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

	namedWindow("亮度分量", WINDOW_NORMAL);
	imshow("亮度分量", V);
	*/

#if 0
    Mat dst = src.clone();
    Mat HT_OpenCV;
    Mat HT_GO;
    Mat AHE_GO;
    Mat CLHE_GO;
    Mat CLAHE_Without_Interpolation;
    Mat CLAHE_OpenCV;
    Mat CLAHE_GO;
    Mat matInter;

    //OpenCV HT方法
    //cv::equalizeHist(src,HT_OpenCV);

    //GO HT方法
    //HT_GO = eaualizeHist_GO(src);

    //GO AHE方法
    AHE_GO = aheGO(src);
	//Mat channel0[3];
	//split(src, channel0);
	//merge(channel0, 3, AHE_GO);

    //GO CLHE方法
    //CLHE_GO = clheGO(src);

    //clahe不计算差值
    //CLAHE_Without_Interpolation = claheGoWithoutInterpolation(src);

    //OpenCV CLAHE 方法
    //Ptr<cv::CLAHE> clahe = createCLAHE();//默认参数
    //clahe->apply(src, CLAHE_OpenCV);

    //GO CLAHE方法
	CLAHE_GO = claheGO(src);

	Mat sub = CLAHE_GO - src;
	//Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));	//可以修改卷积核大小来增加腐蚀效果，越大腐蚀越强
	//erode(sub, sub, kernel);

    //结果显示
    namedWindow("原始图像", WINDOW_NORMAL);
    imshow("原始图像",src);
//    namedWindow("OpencvHT", WINDOW_NORMAL);
//    imshow("OpencvHT",HT_OpenCV);
//    namedWindow("GOHT", WINDOW_NORMAL);
//    imshow("GOHT",HT_GO);
//    namedWindow("GOAHE", WINDOW_NORMAL);
//    imshow("GOAHE",AHE_GO);
//    namedWindow("GOCLHE", WINDOW_NORMAL);
//    imshow("GOCLHE",CLHE_GO);
    namedWindow("CLAHE", WINDOW_NORMAL);
    imshow("CLAHE",CLAHE_GO);

    namedWindow("SUB", WINDOW_NORMAL);
    imshow("SUB", sub);
#endif

    waitKey();
	destroyAllWindows();

    return 0;
}

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

Mat eaualizeHist_GO(Mat src)
{
	int width = src.cols;
	int height = src.rows;
	Mat HT_GO = src.clone();
	int tmp[256] = { 0 };
	float C[256] = { 0.0 };
	int total = width * height;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int index = src.at<uchar>(i, j);
			tmp[index] ++;
		}
	}
	//计算累积函数  
	for (int i = 0; i < 256; i++) {
		if (i == 0)
			C[i] = 1.0f * tmp[i] / total;
		else
			C[i] = C[i - 1] + 1.0f * tmp[i] / total;
	}
	//这里的累积函数分配的方法非常直观高效
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int index = src.at<uchar>(i, j);
			HT_GO.at<uchar>(i, j) = C[index] * 255;
		}
	}
	return HT_GO;
}

Mat aheGO(Mat src, int _step)
{
	Mat AHE_GO = src.clone();
	int block = _step;
	int width = src.cols;
	int height = src.rows;
	int width_block = width / block; //每个小格子的长和宽
	int height_block = height / block;
	//存储各个直方图  
	int tmp2[8 * 8][256] = { 0 };
	float C2[8 * 8][256] = { 0.0 };
	//分块
	int total = width_block * height_block;
	for (int i = 0; i < block; i++)
	{
		for (int j = 0; j < block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//遍历小块,计算直方图
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}
			//计算累积分布直方图  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0)
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				else
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
			}
		}
	}
	//将统计结果写入
	for (int i = 0; i < block; i++)
	{
		for (int j = 0; j < block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//遍历小块,计算直方图
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					//结果直接写入AHE_GO中去
					AHE_GO.at<uchar>(jj, ii) = C2[num][index] * 255;
				}
			}
		}
	}
	return AHE_GO;
}

//这里是在全局直方图加入“限制对比度”方法
Mat clheGO(Mat src, int _step)
{
	int width = src.cols;
	int height = src.rows;
	Mat CLHE_GO = src.clone();
	int tmp[256] = { 0 };
	float C[256] = { 0.0 };
	int total = width * height;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = src.at<uchar>(i, j);
			tmp[index] ++;
		}
	}
	/////////////////////////限制对比度计算部分,注意这个地方average的计算不一定科学
	int average = width * height / 255 / 64;
	int LIMIT = 4 * average;
	int steal = 0;
	for (int k = 0; k < 256; k++)
	{
		if (tmp[k] > LIMIT) {
			steal += tmp[k] - LIMIT;
			tmp[k] = LIMIT;
		}
	}
	int bonus = steal / 256;
	//hand out the steals averagely  
	for (int k = 0; k < 256; k++)
	{
		tmp[k] += bonus;
	}
	///////////////////////////////////////////
	//计算累积函数  
	for (int i = 0; i < 256; i++) {
		if (i == 0)
			C[i] = 1.0f * tmp[i] / total;
		else
			C[i] = C[i - 1] + 1.0f * tmp[i] / total;
	}
	//这里的累积函数分配的方法非常直观高效
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int index = src.at<uchar>(i, j);
			CLHE_GO.at<uchar>(i, j) = C[index] * 255;
		}
	}
	return CLHE_GO;
}

Mat claheGoWithoutInterpolation(Mat src, int _step)
{
	Mat CLAHE_GO = src.clone();
	int block = _step;//pblock
	int width = src.cols;
	int height = src.rows;
	int width_block = width / block; //每个小格子的长和宽
	int height_block = height / block;
	//存储各个直方图  
	int tmp2[8 * 8][256] = { 0 };
	float C2[8 * 8][256] = { 0.0 };
	//分块
	int total = width_block * height_block;
	for (int i = 0; i < block; i++)
	{
		for (int j = 0; j < block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//遍历小块,计算直方图
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}
			//裁剪和增加操作，也就是clahe中的cl部分
			//这里的参数 对应《Gem》上面 fCliplimit  = 4  , uiNrBins  = 255
			int average = width_block * height_block / 255;
			int LIMIT = 4 * average;
			int steal = 0;
			for (int k = 0; k < 256; k++)
			{
				if (tmp2[num][k] > LIMIT) {
					steal += tmp2[num][k] - LIMIT;
					tmp2[num][k] = LIMIT;
				}
			}
			int bonus = steal / 256;
			//hand out the steals averagely  
			for (int k = 0; k < 256; k++)
			{
				tmp2[num][k] += bonus;
			}
			//计算累积分布直方图  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0)
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				else
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
			}
		}
	}
	//计算变换后的像素值  
	//将统计结果写入
	for (int i = 0; i < block; i++)
	{
		for (int j = 0; j < block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//遍历小块,计算直方图
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					//结果直接写入AHE_GO中去
					CLAHE_GO.at<uchar>(jj, ii) = C2[num][index] * 255;
				}
			}
		}

	}
	return CLAHE_GO;
}

Mat claheGO(Mat src, int _step)
{
	Mat CLAHE_GO = src.clone();
	int block = _step;//pblock
	int width = src.cols;
	int height = src.rows;

	//每个小格子的长和宽
	int width_block = width / block;
	int height_block = height / block;

	//存储各个直方图  
	int tmp2[8 * 8][256] = { 0 };
	float C2[8 * 8][256] = { 0.0 };

	//分块
	int total = width_block * height_block;
	for (int i = 0; i < block; i++)
	{
		for (int j = 0; j < block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j; //块序号
			//遍历小块,计算直方图
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}

			//裁剪和增加操作，也就是clahe中的cl部分
			//这里的参数 对应《Gem》上面 fCliplimit  = 4  , uiNrBins  = 255
			int average = width_block * height_block / 255;
			//关于参数如何选择，需要进行讨论。不同的结果进行讨论
			//关于全局的时候，这里的这个cl如何算，需要进行讨论 
			int LIMIT = 40 * average;
			int steal = 0;
			for (int k = 0; k < 256; k++)
			{
				if (tmp2[num][k] > LIMIT) {
					steal += tmp2[num][k] - LIMIT;
					tmp2[num][k] = LIMIT;
				}
			}
			int bonus = steal / 256;
			//hand out the steals averagely  
			for (int k = 0; k < 256; k++)
			{
				tmp2[num][k] += bonus;
			}

			//计算累积分布直方图  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0)
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				else
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
			}
		}
	}
	//计算变换后的像素值  
	//根据像素点的位置，选择不同的计算方法  
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			//four coners  
			if (i <= width_block / 2 && j <= height_block / 2)
			{
				int num = 0;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i <= width_block / 2 && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * (block - 1);
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j <= height_block / 2) {
				int num = block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			//four edges except coners  
			else if (i <= width_block / 2)
			{
				//线性插值  
				int num_i = 0;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2)) {
				//线性插值  
				int num_i = block - 1;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j <= height_block / 2) {
				//线性插值  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = 0;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j >= ((block - 1)*height_block + height_block / 2)) {
				//线性插值  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = block - 1;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//双线性插值
			else {
				int num_i = (i - width_block / 2) / width_block;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				int num3 = num1 + block;
				int num4 = num2 + block;
				float u = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float v = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				CLAHE_GO.at<uchar>(j, i) = (int)((u*v*C2[num4][CLAHE_GO.at<uchar>(j, i)] +
					(1 - v)*(1 - u)*C2[num1][CLAHE_GO.at<uchar>(j, i)] +
					u * (1 - v)*C2[num2][CLAHE_GO.at<uchar>(j, i)] +
					v * (1 - u)*C2[num3][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//最后这步，类似高斯平滑
			CLAHE_GO.at<uchar>(j, i) = CLAHE_GO.at<uchar>(j, i) + (CLAHE_GO.at<uchar>(j, i) << 8) + (CLAHE_GO.at<uchar>(j, i) << 16);
		}
	}
	return CLAHE_GO;
}

cv::Mat Gray2RGB(cv::Mat img, double dp, double minDist, double param1, double param2, int min_radius, int max_radius) {
	//(cv::Mat img, double minDist, int min_radius, int max_radius)
	if (img.empty())
	{
		cout << "can not open image " << endl;
		return img;
	}
	Mat cimg;
	medianBlur(img, img, 5);
	cvtColor(img, cimg, CV_GRAY2RGB);
	vector<Vec3f> circles;
	HoughCircles(img      //InputArray 
		, circles  //OutputArray
		, CV_HOUGH_GRADIENT  //int method
		, 1//dp              //double       dp=1   1 ... 20
		, minDist         //double minDist=10 log 1...1000
		, 100//param1          //double  param1=100
		, 30//param2          //double  param2=30  10 ... 50
		, min_radius      //int  minRadius=1   1 ... 500
		, max_radius      //int  maxRadius=30  1 ... 500
	);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		circle(cimg, Point(c[0], c[1]), c[2], Scalar(255, 0, 0), 3, CV_AA);
		circle(cimg, Point(c[0], c[1]), 2, Scalar(0, 255, 0), 3, CV_AA);
	}
	return cimg;
}



