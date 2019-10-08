#include "header.h"

//如果图像太大缩小一半
Mat mynarrow(Mat img)
{
	Mat dst ;//读出一个图
	if(img.rows*img.cols>2400*1200)
		resize(img,dst,Size(),0.5,0.5); 
	else
		dst=img.clone();
	return dst;
}