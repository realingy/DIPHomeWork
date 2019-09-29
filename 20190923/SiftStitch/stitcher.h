#ifndef __STITCHER_H__ 
#define __STITCHER_H__ 

#include <QObject>
#include <QMutex>
#include <QSemaphore>
#include <QTime>
#include <QCoreApplication>
#include <QEventLoop>
#include <QDebug>
#include <QMetaType>

#include "object.h"

#include "opencv2/core.hpp"  
#include "opencv2/features2d.hpp"  
#include "opencv2/xfeatures2d.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/imgproc.hpp"   
#include "opencv2/core/utility.hpp"  
#include "opencv2/ml.hpp" 
#include "opencv2/calib3d.hpp"  
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace ml;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

class Stitcher : public QObject
{
    Q_OBJECT
public:
    explicit Stitcher(QObject *parent = 0);

signals:
	void sig0(const Mat & graySrc);

public:
	void CalcCorners(const Mat & H, const Mat & src);
	void CalcROICorners(const Mat& H, const Rect & roi);
	Mat stitchTwo(Mat & img1, Mat & img2);
	Mat doStitchTwo(Mat & img1, Mat & img2);
	void getFiles(cv::String dir);
	void updateROI();
	Mat Optimize(Mat& img);
	void stitch();


private:
	Rect roi_;
	int width_;
	int height_;
	four_corners_t corners_;
	four_corners_t cornersroi_;
	vector<Mat> images_;
	vector<cv::String> paths_;

	Object * obj_;

	vector<KeyPoint> key2_;
	Mat key_right_;

};

#endif // __STITCHER_H__ 
