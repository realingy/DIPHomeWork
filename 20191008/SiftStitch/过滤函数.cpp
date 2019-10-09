#include "header.h"

//用单应性过滤匹配
bool refineMatchesWithHomography(const std::vector<cv::KeyPoint> & queryKeypoints,      
    const std::vector<cv::KeyPoint> & trainKeypoints,       
    float reprojectionThreshold,
    std::vector<cv::DMatch> & matches,      
    std::vector<cv::Mat> & homographys
	)    
{
	cv::Mat homography;
    const int minNumberMatchesAllowed = 4;
    if (matches.size() < minNumberMatchesAllowed)
        return false;

    // 为cv::findHomography准备数据
    std::vector<cv::Point2f> queryPoints(matches.size());
    std::vector<cv::Point2f> trainPoints(matches.size());

	//std::vector<cv::Point2f> queryPoints, trainPoints;

    for (size_t i = 0; i < matches.size(); i++)
    {
        queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
        trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
    }

    // 查找单应矩阵并获取内点掩码    
    std::vector<unsigned char> inliersMask(matches.size());
    homography = findHomography(queryPoints, trainPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
	Mat homo;
	invert(homography, homo);
	homographys.push_back(homo);
	cout << "透视变换矩阵： " << "\n" << homo << endl;

    std::vector<cv::DMatch> inliers;
    for (size_t i=0; i<inliersMask.size(); i++)
    {
		if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    matches.swap(inliers);

    //Mat homoShow;
    //drawMatches(src,queryKeypoints,frameImg,trainKeypoints,matches,homoShow,Scalar::all(-1),CV_RGB(255,255,255),Mat(),2);
    //imshow("homoShow",homoShow);
    return matches.size() > minNumberMatchesAllowed;
  
}  


