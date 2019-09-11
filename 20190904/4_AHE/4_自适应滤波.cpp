#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "path.h"

using namespace cv;
using namespace std;

#define CV_ROI_ELEM(src,vector,m,n,ks)  \
{                                      \
    uchar* kn;                         \
    int st0=src.step[0];\
    int st1=src.step[1];\
    for(int k=0;k<(ks);k++)            \
    {                                  \
        for(int s=0;s<(ks);s++)        \
        {                              \
            kn =src.data+(k+m)*st0+(s+n)*st1;   \
            vector.push_back(*kn);              \
        }                                       \
    }                                           \
}

#define CV_MAT_ELEM2(src,dtype,y,x) \
    (dtype*)(src.data+src.step[0]*(y)+src.step[1]*(x))

//自适应滤波
void selfAdaptiveFilter(Mat&src, Mat&dst, int kernal_size)
{
	//CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8U);
	if (dst.empty())
	{
		dst.create(src.rows, src.cols, CV_8UC1);
	}
	uchar* pdst = dst.data;
	uchar Zmin, Zmax, Zmed, Zxy;
	int step0 = src.step[0];
	int step1 = src.step[1];
	for (int i = kernal_size / 2; i < src.rows - kernal_size / 2; i++)
	{
		for (int j = kernal_size / 2; j < src.cols - kernal_size / 2; j++)
		{
			int ks = 3;//kernal_size;
			int count = 0;
			Zxy = *CV_MAT_ELEM2(src, uchar, i, j);//Sxy覆盖区域的中心点像素值，即锚点像素值
			vector<uchar> v;//将模板覆盖区域的像素，压入矢量v中
			do {
				if (count == 0)
				{//获取模板ks*ks覆盖区域的像素，压入矢量v中
					CV_ROI_ELEM(src, v, i - ks / 2, j - ks / 2, ks);
				}
				else
				{
					/****************下面的for循环，将外扩的四个边的像素添加到v中**************/
					uchar* p = src.data + (i - ks / 2)*step0 + (j - ks / 2)*step1;
					for (int u = 0; u < ks; u++)
					{
						v.push_back(*(p + u * step1));//向外扩展的四个边的上边
						v.push_back(*(p + (ks - 1)*step0 + u * step1));//向外扩展的四个边的下边
						if (u != 0 && u != ks - 1)
						{
							v.push_back(*(p + u * step0));//向外扩展的四个边的左边
							v.push_back(*(p + u * step0 + (ks - 1)*step1));//向外扩展的四个边的右边
						}
					}
				}

				//对v的元素排序
				//排序后，Sxy覆盖区域内，最大值为Zmax=v[v.size-1],最小值为Zmin=v[0]
				std::sort(v.begin(), v.end());
				Zmin = v[0], Zmax = v[v.size() - 1], Zmed = v[ks*ks / 2];
				pdst = CV_MAT_ELEM2(dst, uchar, i, j);
				if (Zmin < Zmed&&Zmed < Zmax)
				{
					if (Zmin < Zxy&&Zxy < Zmax)
					{
						*pdst = Zxy; break;
					}
					else
					{
						*pdst = Zmed; break;
					}
				}
				else
				{
					ks += 2;
				}
				count++;
			} while (ks <= kernal_size);

			*pdst = Zmed;
		}
	}
	imshow("selfAdaptiveFilter", dst);
}

int main()
{
	Mat src = imread(MediaPath + "test.bmp");
	imshow("原图", src);
	cout << src.channels() << endl;

	Mat dst = Mat(src.size(), src.type());
	selfAdaptiveFilter(src, dst, 8);

	waitKey(0);

	return 0;
}
