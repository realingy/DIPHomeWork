#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
//#include "path.h"
#include <string>
#include<ctime>
#include <io.h>

using namespace std;
using namespace cv;

bool try_use_gpu = true;
vector<Mat> imgs;
string result_name = "dst1.jpg";
string MediaPath = "images";

vector<Mat> read_images_in_folder(cv::String pattern);

int main(int argc, char * argv[])
{
	cout << "stitch start!\n";
	time_t begin = clock();

	imgs = read_images_in_folder(MediaPath);
	//cout << "###" << imgs.size() << endl;

    Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
    // 使用stitch函数进行拼接
    Mat pano;
    Stitcher::Status status = stitcher.stitch(imgs, pano);
    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return -1;
    }
    imwrite(result_name, pano);
    Mat pano2 = pano.clone();

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "interval: " << interval << endl;

	cout << "stitch end!\n";

    // 显示源图像，和结果图像
    //imshow("全景图像", pano);
    if (waitKey() == 27)
        return 0;
}

vector<Mat> read_images_in_folder(cv::String pattern)
{
	vector<cv::String> paths;
	glob(pattern, paths, false);

	vector<Mat> images;
	//size_t count = fn.size(); //number of png files in images folder
	//for (size_t i = 0; i < count; i++)
	Mat img;
	for ( auto path : paths )
	{
		img = imread(path);
		images.push_back(img);
		//imshow("img", imread(fn[i]));
		//waitKey(1000);
	}
	return images;
}

