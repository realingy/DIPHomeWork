#include <opencv2/core/core.hpp>

using namespace cv;

class CPointGloble: public Point
{
public:
	CPointGloble();
	~CPointGloble();
public:
	Point m_ptOrigin; //目标图片上的点
	Point m_ptCenter; //圆心
	int m_R; //圆的半径
private:
	double m_D; //点和圆心的距离
public:
	bool PtIsInRound(void); //点是否在选定的圆内
	void GlobleToPt(void);  // 球面转到平面上
private:
	double PtDistance(Point pt1,Point pt2);  //计算两点的距离
	double GetAngleFromPt(Point ptStart,Point ptEnd);  //获得和X轴的角度
	Point GetPtFromPolar(double length,double angle);   //获取X Y坐标
	double GetLength();  //获取长度
};