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

    
void Object::slotDetectAndCompute(const Mat & img)
{
//	Mat describor;
//	vector<KeyPoint> keys;
	Ptr<SIFT> sift = SIFT::create(15000);
	
	namedWindow("ttttt", WINDOW_NORMAL);
	imshow("ttttt", img);
	//sift->detect(img, keys);
	sift->detectAndCompute(img, Mat(), keys_, describor_);

//	emit sig1();
}

