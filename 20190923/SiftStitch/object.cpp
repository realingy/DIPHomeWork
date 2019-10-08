#include "object.h"
#include <QDebug>

Object::Object(QObject *parent)
    : QObject(parent)
{
	thread = new QThread();
	moveToThread(thread);
	thread->start();
}

Object::~Object()
{
	thread->quit();
	thread->wait();
	delete thread;
}

    
//void Object::slotDetectAndCompute(int height, int width, const uchar * img)
void Object::slotDetectAndCompute(const Mat & img)
{
	time_t begin = clock();

	Ptr<SIFT> sift = SIFT::create(15000);
	
	sift->detectAndCompute(img, Mat(), keys_, describor_);

#if 0
	//Mat dst;
	//dst.data = (uchar *)img;
	//Mat dst = Mat(height, width, CV_8UC3, (void *)img);
	Mat dst = Mat(height, width, CV_8UC1, (void *)img);
//	namedWindow("dst", WINDOW_NORMAL);
//	imshow("dst", dst);
	imwrite("dst.png", dst);

	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "$$$$interval: " << interval << endl;

	//sift->detectAndCompute(img, Mat(), keys_, describor_);
	sift->detectAndCompute(dst, Mat(), keys_, describor_);
#endif
}

