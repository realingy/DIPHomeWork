#include "object.h"
#include <QDebug>

Object::Object(QObject *parent)
    : QObject(parent)
{

}

/*
void Object::work()
{
	Mat describor;
	vector<KeyPoint> keys;
	Ptr<SIFT> sift = SIFT::create(15000);

	sift->detectAndCompute(img_, Mat(), keys, describor);

	emit sigResult(describor, keys);
}
*/

