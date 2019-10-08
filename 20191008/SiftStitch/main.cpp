#include "header.h"

vector<string> image_names; // image_names[i]表示第i个图像的名称
string dir = "images";

vector<Mat> getFiles(cv::String dir);
extern bool extract_features(
	vector<string> image_names,
	vector<vector<KeyPoint>>& image_keypoints,
	vector<Mat>& image_descriptor//,
	//vector<vector<Vec3b>>& image_colors
);
extern void match_features2(vector<Mat> image_descriptor, vector<vector<DMatch>>& image_matches);
extern bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,
	const std::vector<cv::KeyPoint>& trainKeypoints,
	float reprojectionThreshold,
	std::vector<cv::DMatch>& matches,      
	std::vector<cv::Mat> & homographys
);
extern Mat mynarrow(Mat img);
extern void get_match_points(
	vector<KeyPoint> keypoints1,
	vector<KeyPoint> keypoints2,
	vector<DMatch> matches,
	vector<Point2f>& points1,
	vector<Point2f>& points2
);

int main()
{
	getFiles(dir);

	/* 特征点的提取与匹配 	*/
 
    //LoadImageNamesFromFile("list0.txt",image_names);//从list.txt文件装载图像文件名
 
	vector<vector<KeyPoint>> image_keypoints; // image_keypoints[i]表示第i个图像的特征点
	vector<Mat> image_descriptor; // image_descriptor[i]表示第i个图像的特征向量描述符
	//vector<vector<Vec3b>> image_colors; // image_colors[i]表示第i个图像特征点的颜色
	vector<vector<DMatch>> image_matches; // image[i]表示第i幅图像和第i+1幅图像特征点匹配的结果
	vector<Mat> homos; // homographys[i]表示第i幅图像和第i+1幅图像透视变换矩阵

	time_t begin = clock();
	extract_features(image_names, image_keypoints, image_descriptor/*, image_colors*/); // 提取特征点
	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "extract features interval: " << interval << endl;

	begin = clock();
	match_features2(image_descriptor, image_matches); // 特征点匹配
	end = clock();
	interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "match features interval: " << interval << endl;

	//gms_match_features(image_keypoints,img0.size(),image_matches);
 
	//单应性过滤特征点
	cout << "refine matches with homography!" << image_matches.size() << endl;
	for (int i=0; i<image_matches.size(); i++)
	{
		refineMatchesWithHomography(image_keypoints[i], image_keypoints[i+1], 1.0, image_matches[i], homos);    
	}	
	image_descriptor.swap(vector<Mat>());//匹配完清除内存
 
	Mat img0 = imread(image_names[0]);//读出一个图
	img0 = mynarrow(img0);//如果太大缩小一点。（>2400*1200的）
 
	//显示匹配
	//for (unsigned int i=0;i<image_matches.size ();i++)
	//{
	//	Mat img1 = imread(image_names[i]);
	//	Mat img2 = imread(image_names[i+1]);//读出一个图
 
	//	Mat show = DrawInlier(img1, img2, image_keypoints[i], image_keypoints[i+1], image_matches[i], 1);
	//	imshow("匹配图", show);
	//	char wname[255];
	//	sprintf(wname,"met%d.jpg",i);
	//	imwrite(String(wname),show);
 
 
	//	waitKey();
	//}
 
	vector<cv::Point2f> position_da; // position_da[i]表示第i个图像在大图中的位置(左上角)
	Point2f position_s=Point2f(0,0);
	position_da.push_back (position_s); // 第1个图像为原点
 
	cout << "find position of every image!\n";
	for (unsigned int i=0;i<image_matches.size ();i++)
	{
		if(image_matches[i].size()==0)
			break;//如果无匹配点，则后面的就取消了
 
		//得到匹配点坐标
		vector<Point2f> points1, points2;
		get_match_points (image_keypoints[i], image_keypoints[i+1] ,image_matches[i], points1, points2);
		unsigned int shi=image_matches[i].size();
		shi=(shi>10)?10:shi;//只取前十个
		Point2f a;
		for(unsigned int j=0;j<shi;j++)
		{
			a.x+=points1[j].x-points2[j].x;
			a.y+=points1[j].y-points2[j].y;
		}
		a.x /=shi; 
		a.y /=shi; //取平均值
		// cout << "两个相差："<<a<< endl;
 
		// 在大图的位置
		position_s.x=position_s.x+a.x;
		position_s.y=position_s.y+a.y;
		position_da.push_back(position_s);
		// cout << "当前位置："<<position_s<< endl;
	}

	// 已经用不到了,清除容器并最小化它的容量
	vector<vector<KeyPoint>>().swap(image_keypoints);
 
	// 再计算最小，最大边界
	int xmin=0,xmax=0,ymin=0,ymax=0;
	for (unsigned int i=1;i<position_da.size ();i++)
	{
		xmin=(position_da[i].x<xmin)?position_da[i].x:xmin;
		xmax=(position_da[i].x>xmax)?position_da[i].x:xmax;
		ymin=(position_da[i].y<ymin)?position_da[i].y:ymin;
		ymax=(position_da[i].y>ymax)?position_da[i].y:ymax;
 
	}
	// 计算大图宽高
	int h = img0.rows + ymax-ymin;//拼接图行数(高度)
	int w = img0.cols + xmax-xmin;//拼接图列数（宽度）
	Mat stitch = Mat::zeros(h, w, CV_8UC3);
 
	//再把所有图像放到一个大图中(拼接)
	for (unsigned int i=0;i<position_da.size ();i++)
	{
		img0 = imread(image_names[i]);//读出一个图//左图像
		img0= mynarrow(img0);//如果太大缩小一点。
 
		Mat roi2(stitch, Rect(position_da[i].x-xmin, position_da[i].y-ymin, img0.cols, img0.rows));
        img0(Range(0, img0.rows), Range(0, img0.cols)).copyTo(roi2);
 
	}
 
	namedWindow("拼接结果", WINDOW_NORMAL);
    imshow("拼接结果", stitch);
    imwrite("stitch.png", stitch);
	
	waitKey();

	return 0;


}

vector<Mat> getFiles(cv::String dir)
{
	vector<cv::String> paths;
	glob(dir, paths, false);

	vector<Mat> images;
	for (auto path : paths)
	{
		images.push_back(imread(path));
		image_names.push_back((string)path);
	}

	return images;
}

