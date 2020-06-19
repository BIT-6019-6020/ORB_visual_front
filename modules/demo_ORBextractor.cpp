
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "ORBextractor/myORBextractor.h"

using namespace std;
using namespace cv;
string data_path = "../dates/";
// 设置特征点提取需要的一些参数
int nFeatures = 1000;//图像金字塔上特征点的数量
int nLevels = 8;//图像金字塔层数
float fScaleFactor = 1.2;//金字塔比例因子
int fIniThFAST = 20;//检测fast角点阈值
int fMinThFAST = 8;//最低阈值

int main(int argc, char** argv) {

    //feature extractor
    cv::Mat image = cv::imread(data_path+"test1.png", 0);

    vector<cv::KeyPoint> Keypoints1;
    vector<cv::KeyPoint> Keypoints2;
    Mat descriptors_1, descriptors_2;

    myORB::ORBextractor ORBextractor1 = myORB::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST,
                                                           myORB::ORBextractor::OCTREE, myORB::ORBextractor::ORB);

    myORB::ORBextractor ORBextractor2 = myORB::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST,
                                                            myORB::ORBextractor::CV, myORB::ORBextractor::CV_ORB);

    ORBextractor1(image, cv::Mat(), Keypoints1, descriptors_1);
    ORBextractor2(image, cv::Mat(), Keypoints2, descriptors_2);

    cv::Mat image1, image2, outimg1, outimg2;

    cv::drawKeypoints(image, Keypoints1, outimg1);

    cv::drawKeypoints(image, Keypoints2, outimg2);

    cv::imshow("features1", outimg1);

    cv::imshow("features2", outimg2);


    cv::waitKey(0);
    return 0;

}