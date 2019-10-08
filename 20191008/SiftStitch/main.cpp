#include "header.h"

vector<string> image_names; // image_names[i]��ʾ��i��ͼ�������
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

	/* ���������ȡ��ƥ�� 	*/
 
    //LoadImageNamesFromFile("list0.txt",image_names);//��list.txt�ļ�װ��ͼ���ļ���
 
	vector<vector<KeyPoint>> image_keypoints; // image_keypoints[i]��ʾ��i��ͼ���������
	vector<Mat> image_descriptor; // image_descriptor[i]��ʾ��i��ͼ�����������������
	//vector<vector<Vec3b>> image_colors; // image_colors[i]��ʾ��i��ͼ�����������ɫ
	vector<vector<DMatch>> image_matches; // image[i]��ʾ��i��ͼ��͵�i+1��ͼ��������ƥ��Ľ��
	vector<Mat> homos; // homographys[i]��ʾ��i��ͼ��͵�i+1��ͼ��͸�ӱ任����

	time_t begin = clock();
	extract_features(image_names, image_keypoints, image_descriptor/*, image_colors*/); // ��ȡ������
	time_t end = clock();
	double interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "extract features interval: " << interval << endl;

	begin = clock();
	match_features2(image_descriptor, image_matches); // ������ƥ��
	end = clock();
	interval = double(end - begin) / CLOCKS_PER_SEC;
	cout << "match features interval: " << interval << endl;

	//gms_match_features(image_keypoints,img0.size(),image_matches);
 
	//��Ӧ�Թ���������
	cout << "refine matches with homography!" << image_matches.size() << endl;
	for (int i=0; i<image_matches.size(); i++)
	{
		refineMatchesWithHomography(image_keypoints[i], image_keypoints[i+1], 1.0, image_matches[i], homos);    
	}	
	image_descriptor.swap(vector<Mat>());//ƥ��������ڴ�
 
	Mat img0 = imread(image_names[0]);//����һ��ͼ
	img0 = mynarrow(img0);//���̫����Сһ�㡣��>2400*1200�ģ�
 
	//��ʾƥ��
	//for (unsigned int i=0;i<image_matches.size ();i++)
	//{
	//	Mat img1 = imread(image_names[i]);
	//	Mat img2 = imread(image_names[i+1]);//����һ��ͼ
 
	//	Mat show = DrawInlier(img1, img2, image_keypoints[i], image_keypoints[i+1], image_matches[i], 1);
	//	imshow("ƥ��ͼ", show);
	//	char wname[255];
	//	sprintf(wname,"met%d.jpg",i);
	//	imwrite(String(wname),show);
 
 
	//	waitKey();
	//}
 
	vector<cv::Point2f> position_da; // position_da[i]��ʾ��i��ͼ���ڴ�ͼ�е�λ��(���Ͻ�)
	Point2f position_s=Point2f(0,0);
	position_da.push_back (position_s); // ��1��ͼ��Ϊԭ��
 
	cout << "find position of every image!\n";
	for (unsigned int i=0;i<image_matches.size ();i++)
	{
		if(image_matches[i].size()==0)
			break;//�����ƥ��㣬�����ľ�ȡ����
 
		//�õ�ƥ�������
		vector<Point2f> points1, points2;
		get_match_points (image_keypoints[i], image_keypoints[i+1] ,image_matches[i], points1, points2);
		unsigned int shi=image_matches[i].size();
		shi=(shi>10)?10:shi;//ֻȡǰʮ��
		Point2f a;
		for(unsigned int j=0;j<shi;j++)
		{
			a.x+=points1[j].x-points2[j].x;
			a.y+=points1[j].y-points2[j].y;
		}
		a.x /=shi; 
		a.y /=shi; //ȡƽ��ֵ
		// cout << "������"<<a<< endl;
 
		// �ڴ�ͼ��λ��
		position_s.x=position_s.x+a.x;
		position_s.y=position_s.y+a.y;
		position_da.push_back(position_s);
		// cout << "��ǰλ�ã�"<<position_s<< endl;
	}

	// �Ѿ��ò�����,�����������С����������
	vector<vector<KeyPoint>>().swap(image_keypoints);
 
	// �ټ�����С�����߽�
	int xmin=0,xmax=0,ymin=0,ymax=0;
	for (unsigned int i=1;i<position_da.size ();i++)
	{
		xmin=(position_da[i].x<xmin)?position_da[i].x:xmin;
		xmax=(position_da[i].x>xmax)?position_da[i].x:xmax;
		ymin=(position_da[i].y<ymin)?position_da[i].y:ymin;
		ymax=(position_da[i].y>ymax)?position_da[i].y:ymax;
 
	}
	// �����ͼ���
	int h = img0.rows + ymax-ymin;//ƴ��ͼ����(�߶�)
	int w = img0.cols + xmax-xmin;//ƴ��ͼ��������ȣ�
	Mat stitch = Mat::zeros(h, w, CV_8UC3);
 
	//�ٰ�����ͼ��ŵ�һ����ͼ��(ƴ��)
	for (unsigned int i=0;i<position_da.size ();i++)
	{
		img0 = imread(image_names[i]);//����һ��ͼ//��ͼ��
		img0= mynarrow(img0);//���̫����Сһ�㡣
 
		Mat roi2(stitch, Rect(position_da[i].x-xmin, position_da[i].y-ymin, img0.cols, img0.rows));
        img0(Range(0, img0.rows), Range(0, img0.cols)).copyTo(roi2);
 
	}
 
	namedWindow("ƴ�ӽ��", WINDOW_NORMAL);
    imshow("ƴ�ӽ��", stitch);
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

