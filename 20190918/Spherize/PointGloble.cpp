#include "PointGloble.h"

CPointGloble::CPointGloble()
	:m_R(0),m_ptOrigin(0,0),m_ptCenter(0,0),m_D(0.0)
{
	this->x=-1;
	this->y=-1;  
}

CPointGloble::~CPointGloble()
{

}

double CPointGloble::PtDistance(Point pt1,Point pt2)
{
	int disX,disY;
	disX=pt1.x-pt2.x;
	disY=pt1.y-pt2.y;
	return sqrt(1.0*disX*disX+1.0*disY*disY);
}

double CPointGloble::GetAngleFromPt(Point ptStart,Point ptEnd)
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
		if (disX>0)  
			return atan(double(disY)/disX);
		else
			return atan(double(disY)/disX)+CV_PI;
}

Point CPointGloble::GetPtFromPolar(double length,double angle)
{
	double ptY=sin(angle)*length;
	double ptX=cos(angle)*length;

	ptX+=ptX>0?0.5f:-0.5f;
	ptY+=ptY>0?0.5f:-0.5f;

	return Point(int(ptX),int(ptY));
}

bool CPointGloble::PtIsInRound(void)
{
	m_D=PtDistance(m_ptCenter,m_ptOrigin);
	return m_D<=m_R;
}

void CPointGloble::GlobleToPt(void)
{  
	double length=GetLength();
	double angle=GetAngleFromPt(m_ptCenter,m_ptOrigin);
	Point pt=GetPtFromPolar(length,angle);
	this->x=pt.x+m_ptCenter.x;
	this->y=pt.y+m_ptCenter.y;
}

double CPointGloble::GetLength()
{
	double tpAngle=asin(m_D/(double)m_R);
	if (tpAngle<0) tpAngle=-tpAngle;
	double length=tpAngle*m_R/(CV_PI/2);
	return length;
}