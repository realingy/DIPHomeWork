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
	Ptr<SIFT> sift = SIFT::create(15000);
	
	sift->detectAndCompute(img, Mat(), keys_, describor_);
}

