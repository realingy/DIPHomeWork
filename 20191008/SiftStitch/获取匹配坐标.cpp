#include "header.h"

//获取匹配点坐标
/********************************************************************************************************
参数：
keypoints1 第一张图片的特征点; keypoints2 第二张图片的特征点; matches 匹配的结果; (points1[i], points2[i]) 第
i个匹配的特征点对。
功能：
利用两张图片的特征点keypoints1、keypoints2和匹配的结果matches，可以得到两个数组points1和points2，
(points1[i], points2[i])表示第i个匹配的特征点对。
*********************************************************************************************************/
void get_match_points (
	vector<KeyPoint> keypoints1,
	vector<KeyPoint> keypoints2,
	vector<DMatch> matches,
	vector<Point2f>& points1,
	vector<Point2f>& points2
)
{
	for (int i = 0; i < matches.size (); i++)
	{
		points1.push_back (keypoints1[matches[i].queryIdx].pt);
		points2.push_back (keypoints2[matches[i].trainIdx].pt);
	}
}
 