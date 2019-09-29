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
#include "opencv2/features2d.hpp"  
#include "opencv2/xfeatures2d.hpp"  
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class Object : public QObject
{
    Q_OBJECT
public:
    explicit Object(QObject *parent = 0);

signals:
	void sigResult(const Mat & describor, const vector<KeyPoint> & keys);

public slots:
    void slotDetectAndCompute(const Mat & img)
    {
		//img_ = img;
	
		Mat describor;
		vector<KeyPoint> keys;
		Ptr<SIFT> sift = SIFT::create(15000);

		sift->detectAndCompute(img, Mat(), keys, describor);

		emit sigResult(describor, keys);
    }

protected:
    // void work();

private:
	Mat img_;

};

#endif //__OBJECT_H__ 

