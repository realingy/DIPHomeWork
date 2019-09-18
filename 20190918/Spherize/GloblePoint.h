#include <opencv2/core/core.hpp>

using namespace cv;

class CGloblePoint: public Point
{
public:
	CGloblePoint();
	~CGloblePoint();
public:
	Point m_ptOrigin; //ԭ��ͼƬ�ϵĵ�
	Point m_ptCenter; //Բ��
	int m_R; //Բ�İ뾶
private:
	double m_D; //���Բ�ĵľ���
public:
	bool PtIsInRound(void); //���Ƿ���ѡ����Բ��
	void PtToGloble(void);  //ƽ���ת��������
private:
	double PtDistance(Point pt1,Point pt2);  //��������ľ���
	double GetAngleFromPt(Point ptStart,Point ptEnd);  //��ú�X��ĽǶ�
	Point GetPtFromPolar(double length,double angle);   //��ȡX Y����
};