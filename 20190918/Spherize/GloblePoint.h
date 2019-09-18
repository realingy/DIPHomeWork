#include <opencv2/core/core.hpp>

using namespace cv;

class CGloblePoint: public Point
{
public:
	CGloblePoint();
	~CGloblePoint();
public:
	Point m_ptOrigin; //原来图片上的点
	Point m_ptCenter; //圆心
	int m_R; //圆的半径
private:
	double m_D; //点和圆心的距离
public:
	bool PtIsInRound(void); //点是否在选定的圆内
	void PtToGloble(void);  //平面点转到球面上
private:
	double PtDistance(Point pt1,Point pt2);  //计算两点的距离
	double GetAngleFromPt(Point ptStart,Point ptEnd);  //获得和X轴的角度
	Point GetPtFromPolar(double length,double angle);   //获取X Y坐标
};