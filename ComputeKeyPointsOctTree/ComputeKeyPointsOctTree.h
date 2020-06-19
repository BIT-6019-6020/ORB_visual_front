//
// Created by lab on 20-2-8.
//

#ifndef ORB_SLAM2_MODULES_COMPUTEKEYPOINTSOCTTREE_H
#define ORB_SLAM2_MODULES_COMPUTEKEYPOINTSOCTTREE_H
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
const int EDGE_THRESHOLD = 19;
const int nfeatures = 19;
const int iniThFAST = 19;
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const float W = 30;

void computeOrientation(const Mat &image, vector<KeyPoint> &keypoints, const vector<int> &umax);


float IC_Angle(const Mat &image, Point2f pt, const vector<int> &u_max);

void ComputeKeyPointsOctTree(std::vector<Mat> &mvImagePyramid, vector<vector<KeyPoint> > &allKeypoints,
                             int minThFAST,
                             vector<int> &mnFeaturesPerLevel,
                             std::vector<float> &mvScaleFactor,
                             int nlevels, vector<KeyPoint> &keypointsFinal);

vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY,
                                       const int &N, const int &level,
                                       const int nfeatures,
                                       Mat OctTree_image
);







#endif //ORB_SLAM2_MODULES_COMPUTEKEYPOINTSOCTTREE_H
