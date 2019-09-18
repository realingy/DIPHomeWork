#include "GloblePoint.h"

CGloblePoint::CGloblePoint()
	:m_R(0),m_ptOrigin(0,0),m_ptCenter(0,0),m_D(0.0)
{
	this->x=-1;
	this->y=-1;  //初始化当前点的坐标
}

CGloblePoint::~CGloblePoint()
{

}

double CGloblePoint::PtDistance(Point pt1,Point pt2)
{
	int disX,disY;
	disX=pt1.x-pt2.x;
	disY=pt1.y-pt2.y;
	return sqrt(1.0*disX*disX+1.0*disY*disY);
}

double CGloblePoint::GetAngleFromPt(Point ptStart,Point ptEnd)
{
	int disX=ptEnd.x-ptStart.x;
	int disY=ptEnd.y-ptStart.y;

	if (disX==0)
	{
		if (disY>0)
			return CV_PI/2.0;
		else
			return -CV_PI/2.0;
	}
	else
		if (disX>0)   // 发生错误  这里注意 
			return atan(double(disY)/disX);
		else
			return atan(double(disY)/disX)+CV_PI;
}

Point CGloblePoint::GetPtFromPolar(double length,double angle)
{
	double ptY=sin(angle)*length;
	double ptX=cos(angle)*length;

	ptX+=ptX>0?0.5f:-0.5f;
	ptY+=ptY>0?0.5f:-0.5f;

	return Point(int(ptX),int(ptY));
}

bool CGloblePoint::PtIsInRound(void)
{
	m_D=PtDistance(m_ptCenter,m_ptOrigin);
	return m_D<=m_R;
}

void CGloblePoint::PtToGloble(void)
{
	double length=sin(CV_PI*m_D/m_R/2.0)*m_R;
	double angle=GetAngleFromPt(m_ptCenter,m_ptOrigin);
	Point pt=GetPtFromPolar(length,angle);
	this->x=pt.x+m_ptCenter.x;
	this->y=pt.y+m_ptCenter.y;
}