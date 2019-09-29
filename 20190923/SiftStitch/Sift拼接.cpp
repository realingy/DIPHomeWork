#include <iostream>  
#include <ctime>
#include <QApplication>
#include <QWidget>
#include <QSpinBox>
#include <QObject>
#include <QSlider>
#include <QBoxLayout>
#include "stitcher.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	Stitcher *stitcher = new Stitcher();
	stitcher->stitch();

	int ret = a.exec();
	return ret;
}

