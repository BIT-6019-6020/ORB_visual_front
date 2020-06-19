//
// Created by lab on 20-2-8.
//
//#include "../common_include.h"

#include "ComputeKeyPointsOctTree.h"


//const int PATCH_SIZE = 31;
//const int HALF_PATCH_SIZE = 15;
//const int EDGE_THRESHOLD = 19;
//const int nfeatures = 19;
//const int iniThFAST = 19、、





void
ComputeKeyPointsOctTree(std::vector<Mat> &mvImagePyramid, vector<vector<KeyPoint> > &allKeypoints,
                        int minThFAST,
                        vector<int> &mnFeaturesPerLevel,
                        std::vector<float> &mvScaleFactor, int nlevels, vector<KeyPoint> &keypointsFinal) {

    //

    std::vector<int> umax;

    umax.resize(HALF_PATCH_SIZE + 1);
    allKeypoints.resize(nlevels);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));


//    vector<KeyPoint> keypointsFinal;

    for (int level = 0; level < nlevels; ++level) {
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;  //nCols为的方格列数量
        const int nRows = height / W; //nRows为的方格行数量
        const int wCell = ceil(width / nCols); //重塑格子，让格子尽可能铺满，所以格子这次不一定是方格
        const int hCell = ceil(height / nRows); //每次格子变成wCell*hCell像素的了
        Mat grids_image;
        grids_image = mvImagePyramid[level].clone();

        for (int i = 0; i < nRows; i++) {
            const float iniY = minBorderY + i * hCell;  //iniY起始Y
            float maxY = iniY + hCell + 6;   //为什么+6？

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY) {
                maxY = maxBorderY;
//                Point p1(0, maxY);
//                Point p2(mvImagePyramid[level].cols, maxY);
//                line(image, p1, p2, Scalar(255, 0, 0), 1);
            }

            for (int j = 0; j < nCols; j++) {

                const float iniX = minBorderX + j * wCell;

                float maxX = iniX + wCell + 6;

                if (iniX >= maxBorderX - 6)
                    continue;

                if (maxX > maxBorderX) {
                    maxX = maxBorderX;
//                    Point p3(maxX, 0);
//                    Point p4(maxX, mvImagePyramid[level].rows);
//                    line(image, p3, p4, Scalar(255, 0, 0), 1);

                }

                Point p1(0, iniY);
                Point p2(mvImagePyramid[level].cols, iniY);

                Point p3(iniX, 0);
                Point p4(iniX, mvImagePyramid[level].rows);

                line(grids_image, p1, p2, Scalar(255, 0, 0), 1);
                line(grids_image, p3, p4, Scalar(255, 0, 0), 1);   //画格子  白色
                //mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX);
                //polylines(image, pt, npt, 1, 1, Scalar(0, 0, 255),3);
            }
        }

        imshow("grids", grids_image);
        waitKey(1);

        for (int i = 0; i < nRows; i++) {
            const float iniY = minBorderY + i * hCell;  //iniY起始Y
            float maxY = iniY + hCell + 6;   //为什么+6？

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = maxBorderY;

            for (int j = 0; j < nCols; j++) {
                const float iniX = minBorderX + j * wCell;
                float maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = maxBorderX;

                vector<cv::KeyPoint> vKeysCell;
                vector<cv::KeyPoint> vKeysInPic;
                FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                     vKeysCell, iniThFAST, true);

                if (vKeysCell.empty()) {
                    FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell, minThFAST, true);
                }   //如果使用20检测不到那就使用7，降低阈值


//                Mat image;
//                cv::drawKeypoints(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, image);
//                cv::imshow("image", image);
//                cv::waitKey(0);

                cout << "level*************" << level << endl;

                Rect r(iniX, iniY, wCell, hCell);
                rectangle(grids_image, r, Scalar(0, 255, 255), 1);//画黄色方格，已求出特征点的区域
//                    imshow("grids", image);
//                    waitKey(0);



                if (!vKeysCell.empty()) {
                    cv::KeyPoint tmp_vKeysInPic;
                    for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
                         vit != vKeysCell.end(); vit++) {
//                        tmp_vKeysInPic
                        tmp_vKeysInPic.pt.x = (*vit).pt.x + (minBorderX + j * wCell);  //计算图像方格在整副图像中的位置 or:
                        tmp_vKeysInPic.pt.y = (*vit).pt.y + (minBorderY + i * hCell);
                        vKeysInPic.push_back(tmp_vKeysInPic);

                        (*vit).pt.x += (j * wCell);  //计算图像方格在整副图像中的位置 or:
                        (*vit).pt.y += (i * hCell);

                        vToDistributeKeys.push_back(*vit);
                    }
                }

                cv::drawKeypoints(grids_image, vKeysInPic,
                                  grids_image, Scalar(0, 255, 255));
                cv::imshow("image", grids_image);
                cv::waitKey(1);

            }
        }


        vector<KeyPoint> &keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY, mnFeaturesPerLevel[level], level, nfeatures,
                                      mvImagePyramid[level]);
//            mnFeaturesPerLevel is nDesiredFeaturesPer level
        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++) {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
//        Mat image1 = mvImagePyramid[level];

        cv::drawKeypoints(grids_image, keypoints,
                          grids_image, Scalar(255, 0, 255));
        cv::imshow("image1", grids_image);
        cv::waitKey(1);

        for (int level = 0; level < nlevels; ++level) {

            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax); //计算方向
        }


        if (level != 0) {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        keypointsFinal.insert(keypointsFinal.end(), keypoints.begin(), keypoints.end());
    }

}


//    for (int level = 0; level < nlevels; ++level)
//        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax, nfeatures);




