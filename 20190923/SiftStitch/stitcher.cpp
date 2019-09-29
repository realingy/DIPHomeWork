#include "stitcher.h"
#include <QDebug>

#define BORDERWIDTH 500
#define BORDERHEIGHT 50
	
//void CalcCorners(const Mat & H, const Mat & src);
    
Stitcher::Stitcher(QObject *parent)
	: QObject(parent)
	, roi_(0, 0, 0, 0)
	, width_(0)
	, height_(0)
{

}

void Stitcher::CalcCorners(const Mat & H, const Mat & src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	/*
	//左上角(0,0,1)
	corners_.left_top.x = v1[0] / v1[2];
	corners_.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners_.left_bottom.x = v1[0] / v1[2];
	corners_.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners_.right_top.x = v1[0] / v1[2];
	corners_.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners_.right_bottom.x = v1[0] / v1[2];
	corners_.right_bottom.y = v1[1] / v1[2];
	*/
}

void Stitcher::CalcROICorners(const Mat& H, const Rect & roi)
{
	//左上角(roi.x, roi.y, 1)
	double v2[] = { double(roi.x), double(roi.y), 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	/*
	cornersroi_.left_top.x = v1[0] / v1[2];
	cornersroi_.left_top.y = v1[1] / v1[2];

	//右上角(roi.x + roi.width, roi.y, 1)
	v2[0] = roi.x + roi.width;
	v2[1] = roi.y;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi_.right_top.x = v1[0] / v1[2];
	cornersroi_.right_top.y = v1[1] / v1[2];

	//左下角(roi.x, roi.y + roi.height, 1)
	v2[0] = roi.x;
	v2[1] = roi.y + roi.height;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi_.left_bottom.x = v1[0] / v1[2];
	cornersroi_.left_bottom.y = v1[1] / v1[2];

	//右:下角(roi.x, roi.y + roi.height, 1)
	v2[0] = roi.x + roi.width;
	v2[1] = roi.y + roi.height;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;

	cornersroi_.right_bottom.x = v1[0] / v1[2];
	cornersroi_.right_bottom.y = v1[1] / v1[2];
	*/
}

Mat Stitcher::stitchTwo(Mat & img1, Mat & img2)
{
	Mat img1roi = img1(roi_);

	Mat temp = doStitchTwo(img1roi, img2);

	Mat dst;
	int addwidth = temp.cols - roi_.width;
	int addheight = temp.rows - roi_.height;
	copyMakeBorder(img1, dst, 0, addheight, addwidth, 0, 0, Scalar(0, 0, 0));

	temp.copyTo(dst(Rect(roi_.x, roi_.y, temp.cols, temp.rows)));

	//update ROI
	updateROI();

	return dst;
}

Mat Stitcher::doStitchTwo(Mat & img1, Mat & img2)
{
	time_t begin = clock();

	Mat imageSrc;
	Mat imageMatch;

	// size matches
	int width1 = img1.cols;
	int height1 = img1.rows;
	int width2 = img2.cols;
	int height2 = img2.rows;

	int addh = height1 - height2;
	int addw = width1 - width2;
	//cout << "addw: " << addw << "addh: " << addh << endl;

	// make border
	int addtop = 0;
	int addbottom = BORDERHEIGHT;
	int addleft = BORDERWIDTH; //小于420出现拼接模糊
	int addright = 0;
	//copyMakeBorder(img2, imageMatch, addh, addbottom , addleft, addw, 0, Scalar(0, 0, 0));
	copyMakeBorder(img2, imageMatch, addtop, addbottom + addh, addleft+addw, addright, 0, Scalar(0, 0, 0));
	int h = imageMatch.rows * 0.2;
	copyMakeBorder(img1, imageSrc, addtop, addbottom + h, addleft, addright, 0, Scalar(0, 0, 0));

	// Ptr<SIFT> sift; //创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2, 否则即使配置好了还是显示SIFT为未声明的标识符  
	Ptr<SIFT> sift = SIFT::create(15000);

	//Ptr<ORB> sift = ORB::create(8000);
	//sift->setFastThreshold(0);

	//BFMatcher matcher; //实例化一个暴力匹配器
	FlannBasedMatcher matcher; //实例化Flann匹配器
	vector<KeyPoint> key1, key2;
	vector<DMatch> matches;    //DMatch是用来描述匹配好的一对特征点的类，包含这两个点之间的相关信息
							   //比如左图有个特征m，它和右图的特征点n最匹配，这个DMatch就记录它俩最匹配，并且还记录m和n的
							   //特征向量的距离和其他信息，这个距离在后面用来做筛选

	int rows = imageMatch.rows;
	int cols = imageMatch.cols;

	Mat graySrc, grayMatch;
	cvtColor(imageSrc, graySrc, COLOR_BGR2GRAY);
	cvtColor(imageMatch, grayMatch, COLOR_BGR2GRAY);

	//timeCounter("xx", begin);
	//sift->detectAndCompute(imageMatch, Mat(), key1, key_left); //输入图像，输入掩码，输入特征点，输出Mat，存放所有特征点的描述向量
	Mat key_left, key_right;
	sift->detectAndCompute(grayMatch, Mat(), key1, key_left); //输入图像，输入掩码，输入特征点，输出Mat，存放所有特征点的描述向量
	//timeCounter("yy", begin);
	//sift->detectAndCompute(imageSrc, Mat(), key2, key_right); //这个Mat行数为特征点的个数，列数为每个特征向量的尺寸，SURF是64（维）
	sift->detectAndCompute(graySrc, Mat(), key2, key_right); //这个Mat行数为特征点的个数，列数为每个特征向量的尺寸，SURF是64（维）
	//timeCounter("zz", begin);

	//Mat keySrc, keyMatch;
	//drawKeypoints(graySrc, key2, keySrc);//画出特征点
	//imwrite("keySrc.png", keySrc);
	//drawKeypoints(grayMatch, key1, keyMatch);//画出特征点
	//imwrite("keyMatch.png", keyMatch);

	matcher.match(key_right, key_left, matches);             //匹配，数据来源是特征向量，结果存放在DMatch类型里面  

	//sort函数对数据进行升序排列
	sort(matches.begin(), matches.end());     //筛选匹配点，根据match里面特征对的距离从小到大排序
	vector<DMatch> good_matches;
	int ptsPairs = std::min(2000, (int)(matches.size()));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]); //距离最小的500个压入新的DMatch
	}

	//Mat outimg; //drawMatches这个函数直接画出摆在一起的图
	//drawMatches(imageMatch, key1, imageSrc, key2, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
	//namedWindow("特征匹配效果", WINDOW_NORMAL);
	//imshow("特征匹配效果", outimg);

	//计算图像配准点
	vector<Point2f> imagePoints1, imagePoints2;

	for (int i = 0; i < good_matches.size(); i++)
	{
		imagePoints1.push_back(key1[good_matches[i].trainIdx].pt);
		imagePoints2.push_back(key2[good_matches[i].queryIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	// 也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	// Mat homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	// cout << "变换矩阵为：\n" << endl << homo << endl << endl; //输出映射矩阵   

	//计算配准图的四个顶点坐标
	CalcCorners(homo, imageMatch);

	Rect roi = Rect(imageMatch.cols - width_, 0, width_, height_);
	CalcROICorners(homo, roi);

	//图像配准
	Mat imageWrap; // , imageTransform2;
	warpPerspective(imageMatch, imageWrap, homo, Size(imageMatch.cols, imageMatch.rows+h)); //透视变换
	//rectangle(imageWrap, cvPoint(cornersroi.left_bottom.x, cornersroi.left_top.y), cvPoint(cornersroi.right_top.x , cornersroi.right_bottom.y), Scalar(0, 0, 255), 1, 1, 0);


	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageWrap.cols;
	int dst_height = imageWrap.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	for (int i = 1; i < dst_height; ++i) {
		for (int j = 1; j < dst_width; ++j) {
			if(imageWrap.at<Vec3b>(i, j)[0] != 0)
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j);
			else
				dst.at<Vec3b>(i, j) = imageSrc.at<Vec3b>(i, j);
			/*
			if(imageSrc.at<Vec3b>(i, j)[0] != 0 && imageWrap.at<Vec3b>(i, j)[0] == 0)
				dst.at<Vec3b>(i, j) = imageSrc.at<Vec3b>(i, j);
			else if(imageSrc.at<Vec3b>(i, j)[0] == 0 && imageWrap.at<Vec3b>(i, j)[0] != 0)
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j);
			else
				dst.at<Vec3b>(i, j) = imageWrap.at<Vec3b>(i, j) * 0.6 + imageSrc.at<Vec3b>(i, j) * 0.4;
			*/
		}
	}

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "interval: " << interval << endl;

	return dst;
}

vector<Mat> & Stitcher::getFiles(cv::String dir)
{
	glob(dir, paths_, false);

	for ( auto path : paths_ )
	{
		images_.push_back(imread(path));
	}

	return images_;
}

void Stitcher::updateROI()
{
	int startx = min(cornersroi_.left_top.x, cornersroi_.left_bottom.x);
	int starty = min(cornersroi_.left_top.y, cornersroi_.right_top.y);
	int endx = max(cornersroi_.right_top.x, cornersroi_.right_bottom.x);
	int endy = max(cornersroi_.left_bottom.y, cornersroi_.right_bottom.y);
	roi_.x += startx;
	roi_.y += starty;
	roi_.width = endx - startx;
	roi_.height = endy - starty;
}

/*
Mat Stitcher::Optimize(Mat& img)
{

}
*/

void Stitcher::stitch()
{
	cout << "stitch start!\n";
	time_t begin = clock();

	string dir = "images";
	vector<Mat> images = getFiles(dir);

	Mat img0 = images[0];
	Mat img1 = images[1];
	//g_height = img1.rows;
	//g_width = img1.cols;
	//cout << "stitching \"" << paths[1] << "\" ";
	Mat dst = doStitchTwo(img0, img1);
	updateROI();

	/*
	Mat img2 = images[2];
	cout << "stitching \"" << paths[2] << "\" ";
	dst = doStitchTwo(dst, img2);
	updateROI();

	int count = images.size();
	for (int i = 2; i < count; i++)
	{
		//cout << "stitching \"" << paths[i] << "\" ";
		dst = stitcher->stitchTwo(dst, images[i]);
	}
	*/

	//dst = Optimize(dst); // 裁剪

	//rectangle(dst, cvPoint(roi.x, roi.y), cvPoint(roi.x+roi.width, roi.y+roi.height), Scalar(0, 0, 255), 2, 2, 0);
	
	//中值滤波
	Mat res;
	medianBlur(dst, res, 3);

	namedWindow("拼接效果", WINDOW_NORMAL);
	imshow("拼接效果", res);
	imwrite("res.png", res);

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "stitching end, total interval: " << interval << endl;

	waitKey(0);
}


