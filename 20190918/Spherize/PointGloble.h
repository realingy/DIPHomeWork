#include <opencv2/core/core.hpp>

using namespace cv;

class CPointGloble: public Point
{
public:
	CPointGloble();
	~CPointGloble();
public:
	Point m_ptOrigin; //Ŀ��ͼƬ�ϵĵ�
	Point m_ptCenter; //Բ��
	int m_R; //Բ�İ뾶
private:
	double m_D; //���Բ�ĵľ���
public:
	bool PtIsInRound(void); //���Ƿ���ѡ����Բ��
	void GlobleToPt(void);  // ����ת��ƽ����
private:
	double PtDistance(Point pt1,Point pt2);  //��������ľ���
	double GetAngleFromPt(Point ptStart,Point ptEnd);  //��ú�X��ĽǶ�
	Point GetPtFromPolar(double length,double angle);   //��ȡX Y����
	double GetLength();  //��ȡ����
};