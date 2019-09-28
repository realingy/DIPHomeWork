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
	/*	特征点的提取与匹配 	*/
	//vector<string> image_names; // image_names[i]表示第i个图像的名称
 
	//LoadImageNamesFromFile("list.txt",image_names);//从list.txt文件装载图像文件名
	
	vector<string> image_names = getFiles(dir);
 
	vector<vector<KeyPoint>> image_keypoints; // image_keypoints[i]表示第i个图像的特征点
	vector<Mat> image_descriptor; // image_descriptor[i]表示第i个图像的特征向量描述符
	//vector<vector<Vec3b>> image_colors; // image_colors[i]表示第i个图像特征点的颜色
	vector<vector<DMatch>> image_matches; // image[i]表示第i幅图像和第i+1幅图像特征点匹配的结果
	extract_features (image_names, image_keypoints, image_descriptor/*, image_colors*/); // 提取特征点
	match_features2 (image_descriptor, image_matches); // 特征点匹配
 
	image_descriptor.swap(vector<Mat>());//匹配完清除内存
 
	Mat img0 = imread(image_names[0]);//读出一个图
	//gms_match_features(image_keypoints,img0.size(),image_matches);
 
 
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
 
	//查找向应矩阵
	vector<Mat> im_Homography; // im_Homography[i]表示第i+1-->i的单应矩阵
 
	for (unsigned int i=0;i<image_matches.size ();i++)
	{
 
		//单应矩阵
		Mat h12 = myfindHomography(image_matches[i],  image_keypoints[i], image_keypoints[i+1] );
 
 
		Mat h21;
		invert(h12, h21, DECOMP_LU);
		im_Homography.push_back(h21);
 
	}
	vector<vector<KeyPoint>>().swap(image_keypoints);//已经用不到了,清除容器并最小化它的容量
 
	//拼接
	Mat canvas;
	int canvasSize=image_names.size()*1.5;
	unsigned int j=image_names.size();
		j--;
		Mat img2 = imread(image_names[j]);//读出最后的图
 
	for (unsigned int i=0;i<image_matches.size ();i++)
	{
		//从后到前
		Mat img1;
		Mat h21;
		j--;
		if(j==image_matches.size ()-1){//最右图
			h21=im_Homography[j];
			//使用透视变换
			warpPerspective(img2, canvas, h21, Size(img0.cols*canvasSize, img0.rows));
			img1 = imread(image_names[j]);//读出最后的哪个图
			//拼接
			img1.copyTo(canvas(Range::all(), Range(0, img0.cols)));
		}
		else{//其它
			h21=im_Homography[j];
 
			Mat temp2=canvas.clone();        //保存拷贝
			warpPerspective(temp2, canvas, h21, Size(img0.cols*canvasSize, img0.rows));//一起透视变换
			img1 = imread(image_names[j]);//读出当前的哪个图
			img1.copyTo(canvas(Range::all(), Range(0, img0.cols)));//加当前（拼接）
 
		}
		//imshow("拼接图",canvas);
		char wname[255];
		sprintf(wname,"can%d.jpg",i);
		imwrite(String(wname),canvas);
 
		
		waitKey();
	}
	return 0;
}