#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "GloblePoint.h"
#include "PointGloble.h"
#include "path.h"

using namespace cv;

int main()
{  
	Mat Src,Dst;
	Src=imread(MediaPath + "1.jpg");
	if (!Src.data)
	{
		printf("加载失败");
		return -1;
	}
	int nChannels=Src.channels();
    int MinDiameter=Src.cols<Src.rows?Src.cols:Src.rows;  //最小直径
	CGloblePoint ptGloble;
	CPointGloble ptPoint;
	ptGloble.m_ptCenter=Point(Src.cols/2,Src.rows/2);
	ptPoint.m_ptCenter=Point(Src.cols/2,Src.rows/2);
	int Dialt=30;
	imshow("原图像",Src);
	while(1)
	{
       Dst=Mat::zeros(Src.size(),Src.type());
	   int c=waitKey();
	   if (c==27) break;
	   char C=(char)c;
	   if (c=='1') Dialt+=5;
	   else Dialt-=5;
	   ptPoint.m_R=MinDiameter/2-Dialt;
	   for (int y = 0; y < Dst.rows; y++) {
		   for(int x=0;x<Dst.cols;x++)
		   {   
			   ptPoint.m_ptOrigin=Point(x,y);
			   if (ptPoint.PtIsInRound())
			   {
				   ptPoint.GlobleToPt();
				   Dst.at<Vec3b>(y,x)=Src.at<Vec3b>(ptPoint.y,ptPoint.x);
			   }
			   else
				   Dst.at<Vec3b>(y,x)=Src.at<Vec3b>(y,x);
		   }
	   }
	   imshow("变化",Dst);
	}
	   
	std::cout << "###################\n";
	destroyAllWindows();
	Src.release();
	Dst.release();
	return 0;
}