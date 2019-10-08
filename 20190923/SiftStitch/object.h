#ifndef __OBJECT_H__ 
#define __OBJECT_H__ 

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QSemaphore>
#include <QCoreApplication>
#include <QEventLoop>
#include <QDebug>

#include "opencv2/core.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/imgproc.hpp"   
#include "opencv2/features2d.hpp"  
#include "opencv2/xfeatures2d.hpp"  
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class Object : public QObject
{
    Q_OBJECT
public:
    explicit Object(QObject *parent = 0);
    ~Object();

	Mat describor_;
	vector<KeyPoint> keys_;

signals:
	//void sig1(const Mat & describor, const vector<KeyPoint> & keys);
	void sig1();

public slots:
	void slotDetectAndCompute(const Mat & img);

protected:
    // void work();

private:
	Mat img_;
	QThread *thread;

};

#endif //__OBJECT_H__ 

