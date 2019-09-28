#include "header.h"

char *myfgets(char *s, FILE *fp1);
void LoadImageNamesFromFile(char* name, vector<string>& image_names);
bool extract_features(
	vector<string> image_names,
	vector<vector<KeyPoint>>& image_keypoints,
	vector<Mat>& image_descriptor//,
	//vector<vector<Vec3b>>& image_colors
);
void match_features2(vector<Mat> image_descriptor, vector<vector<DMatch>>& image_matches);
Mat myfindHomography(std::vector< DMatch > & good_matches, std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint> & keypoints_2);
//vector<Mat> getFiles(cv::String dir);
vector<string> getFiles(cv::String dir);

//vector<cv::String> paths;
string dir = "images";

int main ()
{
	/*	���������ȡ��ƥ�� 	*/
	//vector<string> image_names; // image_names[i]��ʾ��i��ͼ�������
 
	//LoadImageNamesFromFile("list.txt",image_names);//��list.txt�ļ�װ��ͼ���ļ���
	
	vector<string> image_names = getFiles(dir);
 
	vector<vector<KeyPoint>> image_keypoints; // image_keypoints[i]��ʾ��i��ͼ���������
	vector<Mat> image_descriptor; // image_descriptor[i]��ʾ��i��ͼ�����������������
	//vector<vector<Vec3b>> image_colors; // image_colors[i]��ʾ��i��ͼ�����������ɫ
	vector<vector<DMatch>> image_matches; // image[i]��ʾ��i��ͼ��͵�i+1��ͼ��������ƥ��Ľ��
	extract_features (image_names, image_keypoints, image_descriptor/*, image_colors*/); // ��ȡ������
	match_features2 (image_descriptor, image_matches); // ������ƥ��
 
	image_descriptor.swap(vector<Mat>());//ƥ��������ڴ�
 
	Mat img0 = imread(image_names[0]);//����һ��ͼ
	//gms_match_features(image_keypoints,img0.size(),image_matches);
 
 
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
 
	//������Ӧ����
	vector<Mat> im_Homography; // im_Homography[i]��ʾ��i+1-->i�ĵ�Ӧ����
 
	for (unsigned int i=0;i<image_matches.size ();i++)
	{
 
		//��Ӧ����
		Mat h12 = myfindHomography(image_matches[i],  image_keypoints[i], image_keypoints[i+1] );
 
 
		Mat h21;
		invert(h12, h21, DECOMP_LU);
		im_Homography.push_back(h21);
 
	}
	vector<vector<KeyPoint>>().swap(image_keypoints);//�Ѿ��ò�����,�����������С����������
 
	//ƴ��
	Mat canvas;
	int canvasSize=image_names.size()*1.5;
	unsigned int j=image_names.size();
		j--;
		Mat img2 = imread(image_names[j]);//��������ͼ
 
	for (unsigned int i=0;i<image_matches.size ();i++)
	{
		//�Ӻ�ǰ
		Mat img1;
		Mat h21;
		j--;
		if(j==image_matches.size ()-1){//����ͼ
			h21=im_Homography[j];
			//ʹ��͸�ӱ任
			warpPerspective(img2, canvas, h21, Size(img0.cols*canvasSize, img0.rows));
			img1 = imread(image_names[j]);//���������ĸ�ͼ
			//ƴ��
			img1.copyTo(canvas(Range::all(), Range(0, img0.cols)));
		}
		else{//����
			h21=im_Homography[j];
 
			Mat temp2=canvas.clone();        //���濽��
			warpPerspective(temp2, canvas, h21, Size(img0.cols*canvasSize, img0.rows));//һ��͸�ӱ任
			img1 = imread(image_names[j]);//������ǰ���ĸ�ͼ
			img1.copyTo(canvas(Range::all(), Range(0, img0.cols)));//�ӵ�ǰ��ƴ�ӣ�
 
		}
		//imshow("ƴ��ͼ",canvas);
		char wname[255];
		sprintf(wname,"can%d.jpg",i);
		imwrite(String(wname),canvas);
 
		
		waitKey();
	}
	return 0;
}