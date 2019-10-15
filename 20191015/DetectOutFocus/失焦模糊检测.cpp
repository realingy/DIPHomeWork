#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

bool focusDetect(Mat& img);
vector<Mat> getFiles(cv::String dir);

vector<String> paths;

int main()
{
	//Mat img = imread("test.PNG");
	//Mat img = imread("test2.png");
	//Mat img = imread("1.jpg");
	vector<Mat> files = getFiles("deblur/");
	//for (auto img : files) {
	for (int i = 0; i < files.size(); i++) {
		Mat img = files[i];
		if (false == focusDetect(img)) {
			cout << paths[i] << ": out of focus!\n";
		} else {
			cout << paths[i] << ": in the focus!\n";
		}
	}

	return 0;
}

//简单设定阈值判断是否失焦
bool focusDetect(Mat& img)
{
    clock_t start, end;
    start = clock();
    int diff = 0;
    int diff_thre = 100;
	int diff_sum_thre = 1000;
    for (int i = img.rows / 10; i < img.rows; i += img.rows / 10){
        uchar* ptrow = img.ptr<uchar>(i);
        for (int j = 0; j < img.cols - 1; j++){
            if (abs(ptrow[j + 1] - ptrow[j])>diff_thre)
                diff += abs(ptrow[j + 1] - ptrow[j]);
        }
        //cout << diff << endl;
    }
    end = clock();
    //cout << "time=" << end - start << endl;

    bool res = true;
    if (diff < diff_sum_thre) {
        //cout << "the focus might be wrong!" << endl;
        res = false;
    }

    return res;
}

vector<Mat> getFiles(cv::String dir)
{
	glob(dir, paths, false);

	vector<Mat> images;
	for (auto path : paths)
	{
		Mat img = imread(path);
		images.push_back(img);
	}
	return images;
}

//返回一个与焦距是否对焦成功的一个比例因子
double focus_measure_GRAT(Mat Image)
{
    double threshold = 0;
    double temp = 0;
    double totalsum = 0;
    int totalnum = 0;

    for (int i=0; i<Image.rows; i++)
    {
        uchar* Image_ptr = Image.ptr<uchar>(i);
        uchar* Image_ptr_1 = Image.ptr<uchar>(i+1);
        for (int j=0; j<Image.cols; j++)
        {
            temp = max(abs(Image_ptr_1[j]-Image_ptr[j]), abs(Image_ptr[j+1]-Image_ptr[j]));
            totalsum += temp;
            totalnum += 1;
        }
    }

    double FM = totalsum/totalnum;

    return FM;
}