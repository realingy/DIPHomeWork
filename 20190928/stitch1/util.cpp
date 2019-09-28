#include "header.h"

//extern vector<cv::String> paths;

//读入一行， 并且去掉结尾的换行符（如果有的话）
char *myfgets(char *s, FILE *fp1)
{
	char tmp[255];
 
	fgets(tmp,255, fp1);//读入一行
 
	char *ret=strrchr(tmp, 10);
	if(ret!=NULL){//去掉结尾的换行符（如果有的话）
			*ret='\0';//memset(ret, 0,1);
		 memcpy(s, tmp, strlen(tmp)+1);//包括\0
	
	}
	else
		return "no";//文件最后一行必须有一个换行符（空行）
	return "ok";
 
}
 
//从list.txt文件装载图像文件名
void LoadImageNamesFromFile(char* name,vector<string>& image_names)
{
	FILE *f = fopen(name, "r");
 
	if (f == NULL) {
	    printf("Error opening file List.txt for reading\n");
	    exit(1);
	}
	char s[255];
	while(1)
	{
		if (myfgets(s,f)=="ok")
			image_names.push_back(string(s));
		else
			break;
	}
}
 
/********************************************************************************************************
参数：
image_names[i] 第i个图像的名称; image_keypoints[i] 第i个图像的特征点;
image_descriptor[i] 第i个图像的特征向量(描述子); image_colors[i] 第i个图像特征点的颜色。
功能：
从一系列图像(image_names)中提取出它们的特征点保存在image_keypoints中，特征向量保存在image_descriptor中，
特征点的颜色保存在image_colors中。
*********************************************************************************************************/
bool extract_features (
	vector<string> image_names,
	vector<vector<KeyPoint>>& image_keypoints,
	vector<Mat>& image_descriptor//,
	//vector<vector<Vec3b>>& image_colors
)
{
	//Ptr<Feature2D> sift = xfeatures2d::SIFT::create (); // SIFT特征提取器
		//sift->detectAndCompute (image, noArray (), keypoints, descriptor);
 
	
		
	//Ptr<AKAZE> akaze = AKAZE::create();
		//akaze->detect(image, keypoints, descriptor);
 
			Ptr<ORB> orb = ORB::create(2100);
			orb->setFastThreshold(0);
 
		//int size;
		for (unsigned int k=0;k<image_names.size();k++)
		{
			string name=image_names[k];
			Mat img1 = imread(name);//data/nn_left.jpg
			if (img1.empty()) {
				printf("出现一个错误，没有找到图像：%s\n",name);
				return false;}
 
			// 提取特征并计算特征向量
			vector<KeyPoint> kp1;
			Mat d1;
			cout << "正在检测特征点: " << name << endl;
			
			orb->detectAndCompute(img1, Mat(), kp1, d1);
			image_keypoints.push_back(kp1);
			image_descriptor.push_back (d1);
 
 
			//cout << "保存角点颜色: " << endl;
			//vector<Vec3b> colors;
			//for (unsigned int i = 0; i < kp1.size(); i++)
			//{
			//	Point2f p = kp1[i].pt;
			//	colors.push_back(img1.at<Vec3b>((int)p.y, (int)p.x));
			//}
			//image_colors.push_back(colors);
		}
 
	return true;
}
 
/********************************************************************************************************
参数：
image_descriptor[i] 第i个图像的特征向量; image_matches[i] 第i个特征向量和第i + 1个特征向量匹配的结果。
功能：
对一组特征向量(image_descriptor)两两匹配，将结果保存在image_matches中。
*********************************************************************************************************/
void match_features2 (vector<Mat> image_descriptor, vector<vector<DMatch>>& image_matches)
{
	for (unsigned int i = 0; i < image_descriptor.size () - 1; i++)
	{
		cout << "正在匹配 " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		//match_features1 (image_descriptor[i], image_descriptor[i + 1], matches);
 
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(image_descriptor[i], image_descriptor[i + 1], matches);
 
		cout << "有 " << matches.size() << " 个匹配点" << endl;
 
		image_matches.push_back (matches);
	}
}
 
//利用findHomography函数利用匹配的关键点找出相应的变换：
Mat myfindHomography(std::vector< DMatch > & good_matches,  std::vector<KeyPoint>& keypoints_1,std::vector<KeyPoint> & keypoints_2 )
{
	//-- Localize the object from img_1 in img_2     //在img_2中定位来自img_1的对象
	std::vector<Point2f> obj;    
	std::vector<Point2f> scene;    
	    
	for(unsigned int i = 0; i < good_matches.size(); i++ )    
	{    
	  //-- Get the keypoints from the good matches    //从好的匹配中获取关键点
	  obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );    
	  scene.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );     
	}    
	    
	//两个平面上相匹配的特征点求出变换公式
	Mat H = findHomography( obj, scene, CV_RANSAC );    
	    
	return H;
}
	
//vector<Mat> getFiles(cv::String dir)
vector<string> getFiles(cv::String dir)
{
	vector<cv::String> paths;
	glob(dir, paths, false);

	/*
	vector<Mat> images;
	for (auto path : paths)
	{
		images.push_back(imread(path));
	}
	return images;
	*/

	vector<string> res;
	for (auto path : paths)
	{
		//images.push_back(imread(path));
		res.push_back(string(path));
	}

	return res;

}

