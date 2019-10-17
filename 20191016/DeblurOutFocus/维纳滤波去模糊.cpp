#include <time.h>
#include <stdlib.h>
#include <cxcore.h>   
#include "stdio.h"
#include <string>
#include <fstream>
#include <highgui.h> 
#include <cv.h> 
#include <windows.h>
#include <iostream>       //存储int型变量用32位

using namespace std;
using namespace cv;

#define eps 2.22044604925031e-016;
#define pi 3.1416;
#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

//# pragma comment(linker, "/NODEFAULTLIB:atlthunk.lib")
//# pragma comment(linker, "/NODEFAULTLIB:LIBCMT")
//# pragma comment(linker, "/NODEFAULTLIB:MSVCRTD")

BOOL fourn(double * data/*psrc*/, unsigned long nn[]/*w*/, int ndim/*2*/, int isign);
double CY_sign(double cosphi);
void CYmeshgrid(int xm,int xn,int ym,int yn,CvMat* &X,CvMat* &Y);
void CY_special(double len,double theta,CvMat *&h_last);

void main(int argc, char argv[])
{
	IplImage *RGB = cvLoadImage("original.jpg",-1);
	IplImage* redImage=cvCreateImage(cvGetSize(RGB),IPL_DEPTH_8U,1);    //定义三个通道图像
	IplImage* greenImage=cvCreateImage(cvGetSize(RGB),IPL_DEPTH_8U,1);    
	IplImage* blueImage=cvCreateImage(cvGetSize(RGB),IPL_DEPTH_8U,1);
	cvSplit(RGB,blueImage,greenImage,redImage,NULL);   //通道分割。注意：OpenCV分割成的三个通道参数顺序是：B,G,R
	
	//cvNamedWindow("RGB");
	//cvShowImage("RGB",RGB);

	//处理蓝色通道
	int bHeight = blueImage->height;
	int bLineBytes = blueImage->widthStep;
	int bw = 1;
	int bh = 1;

	//保证离散傅里叶变换的宽度和高度为2的整数幂
	while(bw*2 <= bLineBytes)
	{
		bw = bw*2;
	}

	while(bh*2 <= bHeight)
	{
		bh = bh*2;
	}

	//输入退化图像的长和宽必须为2的整数倍；
	if(bw != (int)bLineBytes)
	{
		return;
	}

	if(bh != (int)bHeight)
	{
		return;
	}

	//用于做FFT的数组
	double startime = (double)getTickCount(); // set the begining time

	// 指向源图像倒数第j行，第i个象素的指针   
	double *fftSrc, *fftKernel, *laplacianKernel;
	fftSrc = new double [bHeight*bLineBytes*2+1];
	fftKernel = new double [bHeight*bLineBytes*2+1];
laplacianKernel = new double [bHeight*bLineBytes*2+1];
unsigned long nn[3] = {0};
nn[1] = bHeight;
nn[2] = bLineBytes; 
LPSTR lpSrc;
unsigned char pixel;
double len = 15;     //模糊图像的模糊长度
double theta = 60;   //模糊图像的模糊角度
CvMat *h_last;
CY_special(len,theta,h_last);
int h_row = h_last->height;
int h_col = h_last->width;
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
// 指向源图像倒数第j行，第i个象素的指针   
lpSrc = (char *)blueImage->imageData + bLineBytes * j + i;

pixel = (unsigned char)*lpSrc;

fftSrc[(2*bLineBytes)*j + 2*i + 1] = (double)pixel;
fftSrc[(2*bLineBytes)*j + 2*i + 2] = 0.0;
laplacianKernel[(2*bLineBytes)*j + 2*i + 1] = 0.0;
laplacianKernel[(2*bLineBytes)*j + 2*i + 2] = 0.0;
if(i < h_col && j < h_row)
{
float h_value = CV_MAT_ELEM(*h_last,float,j,i);
fftKernel[(2*bLineBytes)*j + 2*i + 1] = double(h_value);
}
else
{
fftKernel[(2*bLineBytes)*j + 2*i + 1] = 0.0;
}
fftKernel[(2*bLineBytes)*j + 2*i + 2] = 0.0;
}
}
laplacianKernel[(2*bLineBytes)*0+2*1+1] = 1.0;  //设置拉普拉斯滤波器
laplacianKernel[(2*bLineBytes)*1+2*0+1] = 1.0;
laplacianKernel[(2*bLineBytes)*1+2*1+1] = -4.0;
laplacianKernel[(2*bLineBytes)*0+2*2+1] = 1.0;
laplacianKernel[(2*bLineBytes)*2+2*1+1] = 1.0;

//对源图像进行FFT
fourn(fftSrc,nn,2,1);
//对卷积核图像进行FFT
fourn(fftKernel,nn,2,1);
//对过滤器进行FFT;
fourn(laplacianKernel,nn,2,1);
double a,b,c,d,e,f,norm1,norm2,temp;
double gama = 0.05;
for (int i = 1;i <bHeight*bLineBytes*2;i+=2)
{
a = fftSrc[i];
b = fftSrc[i+1];
c = fftKernel[i];
d = fftKernel[i+1];
e = laplacianKernel[i];
f = laplacianKernel[i+1];
//计算|H(u,v)|*|H(u,v)|+r|C(u,v)*C(u,v)|;
norm1 = c*c + d*d;
norm2 = e*e + f*f;
temp = norm1 + norm2*gama;
if (c*c + d*d > 1e-3)
{
fftSrc[i] = ( a*c + b*d ) /temp;
fftSrc[i+1] = ( b*c - a*d )/temp; 
}
}

//对结果图像进行反FFT
fourn(fftSrc,nn,2,-1);
//确定归一化因子  
//图像归一化因子
double MaxNum;
MaxNum = 0.0;
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
fftSrc[(2*bLineBytes)*j + 2*i + 1] = 
sqrt(fftSrc[(2*bLineBytes)*j + 2*i + 1] * fftSrc[(2*bLineBytes)*j + 2*i + 1]/
+fftSrc[(2*bLineBytes)*j + 2*i + 2] * fftSrc[(2*bLineBytes)*j + 2*i + 2]);
if( MaxNum < fftSrc[(2*bLineBytes)*j + 2*i + 1])
MaxNum = fftSrc[(2*bLineBytes)*j + 2*i + 1];
}
}
//转换为图像
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
// 指向源图像倒数第j行，第i个象素的指针   
lpSrc = (char *)blueImage->imageData + bLineBytes * j + i;

*lpSrc = (unsigned char) (fftSrc[(2*bLineBytes)*j + 2*i + 1]*255.0/MaxNum);
}
}
//cvNamedWindow("blueImageDeblur");
//cvShowImage("blueImageDeblur",blueImage);

//处理绿色通道
bLineBytes = greenImage->widthStep;
bw = 1;
bh = 1;
//保证离散傅里叶变换的宽度和高度为2的整数幂
while(bw*2 <= bLineBytes)
{
bw = bw*2;
}
while(bh*2 <= bHeight)
{
bh = bh*2;
}
//输入退化图像的长和宽必须为2的整数倍；
if(bw != (int)bLineBytes)
{
return;
}
if(bh != (int)bHeight)
{
return;
}

//用于做FFT的数组
// 指向源图像倒数第j行，第i个象素的指针   
fftSrc = new double [bHeight*bLineBytes*2+1];
fftKernel = new double [bHeight*bLineBytes*2+1];
laplacianKernel = new double [bHeight*bLineBytes*2+1];
CvMat *h_last2;
CY_special(len,theta,h_last2);
h_row = h_last2->height;
h_col = h_last2->width;
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
// 指向源图像倒数第j行，第i个象素的指针   
lpSrc = (char *)greenImage->imageData + bLineBytes * j + i;

pixel = (unsigned char)*lpSrc;

fftSrc[(2*bLineBytes)*j + 2*i + 1] = (double)pixel;
fftSrc[(2*bLineBytes)*j + 2*i + 2] = 0.0;
laplacianKernel[(2*bLineBytes)*j + 2*i + 1] = 0.0;
laplacianKernel[(2*bLineBytes)*j + 2*i + 2] = 0.0;
if(i < h_col && j < h_row)
{
float h_value = CV_MAT_ELEM(*h_last2,float,j,i);
fftKernel[(2*bLineBytes)*j + 2*i + 1] = double(h_value);
}
else
{
fftKernel[(2*bLineBytes)*j + 2*i + 1] = 0.0;
}
fftKernel[(2*bLineBytes)*j + 2*i + 2] = 0.0;
}
}
laplacianKernel[(2*bLineBytes)*0+2*1+1] = 1.0;  //设置拉普拉斯滤波器
laplacianKernel[(2*bLineBytes)*1+2*0+1] = 1.0;
laplacianKernel[(2*bLineBytes)*1+2*1+1] = -4.0;
laplacianKernel[(2*bLineBytes)*0+2*2+1] = 1.0;
laplacianKernel[(2*bLineBytes)*2+2*1+1] = 1.0;

//对源图像进行FFT
fourn(fftSrc,nn,2,1);
//对卷积核图像进行FFT
fourn(fftKernel,nn,2,1);
//对过滤器进行FFT;
fourn(laplacianKernel,nn,2,1);
for (int i = 1;i <bHeight*bLineBytes*2;i+=2)
{
a = fftSrc[i];
b = fftSrc[i+1];
c = fftKernel[i];
d = fftKernel[i+1];
e = laplacianKernel[i];
f = laplacianKernel[i+1];
//计算|H(u,v)|*|H(u,v)|+r|C(u,v)*C(u,v)|;
norm1 = c*c + d*d;
norm2 = e*e + f*f;
temp = norm1 + norm2*gama;
if (c*c + d*d > 1e-3)
{
fftSrc[i] = ( a*c + b*d ) /temp;
fftSrc[i+1] = ( b*c - a*d )/temp; 
}
}

//对结果图像进行反FFT
fourn(fftSrc,nn,2,-1);
//确定归一化因子  
//图像归一化因子
MaxNum=0;
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
fftSrc[(2*bLineBytes)*j + 2*i + 1] = 
sqrt(fftSrc[(2*bLineBytes)*j + 2*i + 1] * fftSrc[(2*bLineBytes)*j + 2*i + 1]/
+fftSrc[(2*bLineBytes)*j + 2*i + 2] * fftSrc[(2*bLineBytes)*j + 2*i + 2]);
if( MaxNum < fftSrc[(2*bLineBytes)*j + 2*i + 1])
MaxNum = fftSrc[(2*bLineBytes)*j + 2*i + 1];
}
}
//转换为图像
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
// 指向源图像倒数第j行，第i个象素的指针   
lpSrc = (char *)greenImage->imageData + bLineBytes * j + i;

*lpSrc = (unsigned char) (fftSrc[(2*bLineBytes)*j + 2*i + 1]*255.0/MaxNum);
}
}
//cvNamedWindow("greenImageDeblur");
//cvShowImage("greenImageDeblur",greenImage);

//处理绿色通道
bLineBytes = redImage->widthStep;
bw = 1;
bh = 1;
//保证离散傅里叶变换的宽度和高度为2的整数幂
while(bw*2 <= bLineBytes)
{
bw = bw*2;
}
while(bh*2 <= bHeight)
{
bh = bh*2;
}
//输入退化图像的长和宽必须为2的整数倍；
if(bw != (int)bLineBytes)
{
return;
}
if(bh != (int)bHeight)
{
return;
}

//用于做FFT的数组
// 指向源图像倒数第j行，第i个象素的指针   
fftSrc = new double [bHeight*bLineBytes*2+1];
fftKernel = new double [bHeight*bLineBytes*2+1];
laplacianKernel = new double [bHeight*bLineBytes*2+1];
CvMat *h_last3;
CY_special(len,theta,h_last3);
h_row = h_last3->height;
h_col = h_last3->width;
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
// 指向源图像倒数第j行，第i个象素的指针   
lpSrc = (char *)redImage->imageData + bLineBytes * j + i;

pixel = (unsigned char)*lpSrc;

fftSrc[(2*bLineBytes)*j + 2*i + 1] = (double)pixel;
fftSrc[(2*bLineBytes)*j + 2*i + 2] = 0.0;
laplacianKernel[(2*bLineBytes)*j + 2*i + 1] = 0.0;
laplacianKernel[(2*bLineBytes)*j + 2*i + 2] = 0.0;
if(i < h_col && j < h_row)
{
float h_value = CV_MAT_ELEM(*h_last3,float,j,i);
fftKernel[(2*bLineBytes)*j + 2*i + 1] = double(h_value);
}
else
{
fftKernel[(2*bLineBytes)*j + 2*i + 1] = 0.0;
}
fftKernel[(2*bLineBytes)*j + 2*i + 2] = 0.0;
}
}
laplacianKernel[(2*bLineBytes)*0+2*1+1] = 1.0;  //设置拉普拉斯滤波器
laplacianKernel[(2*bLineBytes)*1+2*0+1] = 1.0;
laplacianKernel[(2*bLineBytes)*1+2*1+1] = -4.0;
laplacianKernel[(2*bLineBytes)*0+2*2+1] = 1.0;
laplacianKernel[(2*bLineBytes)*2+2*1+1] = 1.0;

//对源图像进行FFT
fourn(fftSrc,nn,2,1);
//对卷积核图像进行FFT
fourn(fftKernel,nn,2,1);
//对过滤器进行FFT;
fourn(laplacianKernel,nn,2,1);
for (int i = 1;i <bHeight*bLineBytes*2;i+=2)
{
a = fftSrc[i];
b = fftSrc[i+1];
c = fftKernel[i];
d = fftKernel[i+1];
e = laplacianKernel[i];
f = laplacianKernel[i+1];
//计算|H(u,v)|*|H(u,v)|+r|C(u,v)*C(u,v)|;
norm1 = c*c + d*d;
norm2 = e*e + f*f;
temp = norm1 + norm2*gama;
if (c*c + d*d > 1e-3)
{
fftSrc[i] = ( a*c + b*d ) /temp;
fftSrc[i+1] = ( b*c - a*d )/temp; 
}
}

//对结果图像进行反FFT
fourn(fftSrc,nn,2,-1);
//确定归一化因子  
//图像归一化因子
MaxNum=0;
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
fftSrc[(2*bLineBytes)*j + 2*i + 1] = 
sqrt(fftSrc[(2*bLineBytes)*j + 2*i + 1] * fftSrc[(2*bLineBytes)*j + 2*i + 1]/
+fftSrc[(2*bLineBytes)*j + 2*i + 2] * fftSrc[(2*bLineBytes)*j + 2*i + 2]);
if( MaxNum < fftSrc[(2*bLineBytes)*j + 2*i + 1])
MaxNum = fftSrc[(2*bLineBytes)*j + 2*i + 1];
}
}
//转换为图像
for (int j = 0;j < bHeight ;j++)
{
for(int i = 0;i < bLineBytes ;i++)
{
// 指向源图像倒数第j行，第i个象素的指针   
lpSrc = (char *)redImage->imageData + bLineBytes * j + i;

*lpSrc = (unsigned char) (fftSrc[(2*bLineBytes)*j + 2*i + 1]*255.0/MaxNum);
}
}
//cvNamedWindow("redImageDeblur");
//cvShowImage("redImageDeblur",redImage);

IplImage* MergedImage=cvCreateImage(cvGetSize(RGB),IPL_DEPTH_8U,3);  //将分割后的三个通道进行合并
cvMerge(blueImage,greenImage,redImage,0,MergedImage);  

double durationtime = (double)getTickCount() - startime; 
printf("  detection time = %g ms\n", durationtime*1000./cv::getTickFrequency()); //calculate the running time
cvNamedWindow("Merged Image");
cvShowImage("Merged Image",MergedImage);
cvWaitKey();
}

BOOL fourn(double * data/*psrc*/, unsigned long nn[]/*w*/, int ndim/*2*/, int isign)
{
int idim;
unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
double tempi,tempr;
double theta,wi,wpi,wpr,wr,wtemp;

for (ntot=1,idim=1;idim<=ndim;idim++)
ntot *= nn[idim];
nprev=1;
for (idim=ndim;idim>=1;idim--) {
n=nn[idim];
nrem=ntot/(n*nprev);
ip1=nprev << 1;
ip2=ip1*n;
ip3=ip2*nrem;
i2rev=1;
for (i2=1;i2<=ip2;i2+=ip1) {
if (i2  < i2rev) {
for (i1=i2;i1<=i2+ip1-2;i1+=2) {
for (i3=i1;i3<=ip3;i3+=ip2) {
i3rev=i2rev+i3-i2;
SWAP(data[i3],data[i3rev]);
SWAP(data[i3+1],data[i3rev+1]);
}
}
}
ibit=ip2 >> 1;
while (ibit >= ip1 && i2rev > ibit) {
i2rev -= ibit;
ibit >>= 1;
}
i2rev += ibit;
}
ifp1=ip1;
while (ifp1 < ip2) {
ifp2=ifp1 << 1;
theta=isign*6.28318530717959/(ifp2/ip1);
wtemp=sin(0.5*theta);
wpr = -2.0*wtemp*wtemp;
wpi=sin(theta);
wr=1.0;
wi=0.0;
for (i3=1;i3<=ifp1;i3+=ip1) {
for (i1=i3;i1<=i3+ip1-2;i1+=2) {
for (i2=i1;i2<=ip3;i2+=ifp2) {
k1=i2;
k2=k1+ifp1;
tempr=wr*data[k2]-wi*data[k2+1];
tempi=wr*data[k2+1]+wi*data[k2];
data[k2]=data[k1]-tempr;
data[k2+1]=data[k1+1]-tempi;
data[k1] += tempr;
data[k1+1] += tempi;
}
}
wr=(wtemp=wr)*wpr-wi*wpi+wr;
wi=wi*wpr+wtemp*wpi+wi;
}
ifp1=ifp2;
}
nprev *= n;
}
return true;
}
void CY_special(double len,double theta,CvMat *&h_last)
{
len = double(max(len,1.0));
double half = (len - 1)/2;
double phi = theta/180.0*pi;
double cosphi = cos(phi);
double sinphi = sin(phi);
double xsign = CY_sign(cosphi);
double linewdt = 1.0;
double sx = half*cosphi + linewdt*xsign - len*eps;
sx = cvFloor(sx);
double sy = half*sinphi +linewdt - len*eps;
sy = cvFloor(sy);
CvMat *X,*Y;
CYmeshgrid(0,int(sy),0,int(sx),X,Y);
int row = X->height;
int col = X->width;
CvMat *dist2line = cvCreateMat(row,col,CV_32FC1);
CvMat *rad = cvCreateMat(row,col,CV_32FC1);
CvMat *h = cvCreateMat(row,col,CV_32FC1);
cvZero(dist2line);
cvAddWeighted(Y,cosphi,X,-sinphi,0,dist2line);
cvCartToPolar(X,Y,rad,NULL,0);
for(int i = 0;i<row;i++)
{
for(int j = 0;j<col;j++)
{
float temp1 = CV_MAT_ELEM(*rad,float,i,j);
float temp2 = CV_MAT_ELEM(*dist2line,float,i,j);
temp2 = abs(temp2);
if(temp1 >= half && temp2 <= linewdt)
{
float x_value = CV_MAT_ELEM(*X,float,i,j);
float dist_value = CV_MAT_ELEM(*dist2line,float,i,j);
float dist_value1 = (x_value + dist_value*float(sinphi))/float(cosphi);
float x2lastpix_cy1 =  float(half) - abs(dist_value1);
float cy = dist_value*dist_value + x2lastpix_cy1*x2lastpix_cy1;
float cy1 = cvSqrt(cy);
cvmSet(dist2line,i,j,cy1);
}
}
}
for(int i = 0;i<row;i++)
{
for(int j = 0;j<col;j++)
{
float cy2 = CV_MAT_ELEM(*dist2line,float,i,j);
cy2 = float(linewdt) + float(0.00000000001) - abs(cy2);
cvmSet(dist2line,i,j,cy2);
if(cy2<0)
{
cvmSet(dist2line,i,j,0.0);
}
}
}
cvFlip(dist2line,h,1);
cvFlip(h,NULL,0);
int row_h = row + row - 1;
int col_h = col + col - 1;
//CvMat *h_last = cvCreateMat(row_h,col_h,CV_32FC1);
h_last = cvCreateMat(row_h,col_h,CV_32FC1);
cvZero(h_last);
for(int i = 0;i<row;i++)
{
for(int j = 0;j<col;j++)
{
float h_value = CV_MAT_ELEM(*h,float,i,j);
cvmSet(h_last,i,j,h_value);
}
}
//将dist2line中的值赋给扩大后的h矩阵中的相应位置
int p =0;
for(int i = row-1;i<row_h;i++)
{
int q =0;
for(int j = col-1;j<col_h;j++)
{
float h_temp = CV_MAT_ELEM(*dist2line,float,p,q);
cvmSet(h_last,i,j,h_temp);
q++;
}
p++;
}
double yao_value = len*len*eps;
CvScalar sum = cvSum(h_last);
double yao_sum = sum.val[0];
double yaoyao = 1/(yao_value + yao_sum);
cvConvertScale(h_last,h_last,yaoyao,0);
if(cosphi > 0)
{
cvFlip(h_last,NULL,0);
}
}

double CY_sign(double cosphi)
{
double dst_value = 0;
if(cosphi>0)
{
dst_value = 1;
}
else
{
dst_value = -1;
}
return dst_value;
}

void CYmeshgrid(int xm,int xn,int ym,int yn,CvMat* &X,CvMat* &Y)
{
int m=xn-xm+1;
int n=yn-ym+1;
X=cvCreateMat(m,n,CV_32FC1);
Y=cvCreateMat(m,n,CV_32FC1);
float cy2=float(xm);
for(int i=0;i<m;i++)
{
float cy1=float(xm);
for(int j=0;j<n;j++)
{
cvmSet(X,i,j,cy1);
cvmSet(Y,i,j,cy2);
cy1=cy1+1;
}
cy2=cy2+1;
}
}