//
// Created by lab on 20-2-9.
//

#include "ComputeKeyPointsOctTree.h"


class ExtractorNode {
public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
    void DrawGrid(Mat& image);
    void DrawKey(Mat image);
    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;  //以方格中心点为原点分为四个象限
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

void ExtractorNode::DrawGrid(Mat& image){
    Rect r(UL, BR);
    rectangle(image, r, Scalar(255, 255, 0), 1);//画方格，已求出特征点的区域
}

//cv::drawKeypoints(image, allKeypoints[0], image);

void ExtractorNode::DrawKey(Mat image){

    this->DrawGrid(image);
    cv::drawKeypoints(image, this->vKeys, image);
    imshow("key in grides",image);
    waitKey(0);
}


void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4) {
    const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for (size_t i = 0; i < vKeys.size(); i++) {
        const cv::KeyPoint &kp = vKeys[i];
        if (kp.pt.x < n1.UR.x) {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        } else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }  //将关键点分类到四个象限

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;

}


vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY,
                                       const int &N, const int &level,
                                       const int nfeatures,
                                       Mat OctTree_image
) {
    // Compute how many initial nodes
    // 首先计算根结点的个数
    // 默认图像 宽>长
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

    const float hX = static_cast<float>(maxX - minX) / nIni;

    list<ExtractorNode> lNodes;  //list相比于vector最大的优势就是：list是非连续的存储结构，可以进行快速的插入和删除操作
    //方便查找和删除节点

    vector<ExtractorNode *> vpIniNodes;
    vpIniNodes.resize(nIni);

    // 根据原始根结点数量,设置其4个对应的子结点,并将其加入到根节点的队列中
    for (int i = 0; i < nIni; i++) {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());  //关键点总数目

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // 将所有的特征点加入到初始的对应的根结点中
    // 其实这里可以直接用lN
    // odes进行寻址,用指针容器是否要快一点
    for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
        const cv::KeyPoint &kp = vToDistributeKeys[i];

        // 对于宽/长没有超过1.5的图像,所有的特征点都归入了唯一的根结点
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    // 检查结点中若有空节点则去掉,若结点包含的特征点为1则表示其不再分
    while (lit != lNodes.end()) {
        if (lit->vKeys.size() == 1) {
            lit->bNoMore = true;
            lit++;
        } else if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, ExtractorNode *> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    // 开始循环,循环的目的是按照四叉树细分每个结点,直到每个结点不可分为止,不可分包含两种情况,结点所含特征点为1或>1,之所以要用四叉树,
    // 就是要快速找到扎堆的点并取其最大响应值的点作为最终的特征点
    while (!bFinish) {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        // nToExpand代表还可分的结点个数
        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        // 一次循环下来,将当前lNode中的根结点全部细分成子结点,并将子结点加入到lNode的前部且不会被遍历到,删除对应的根结点
        // 注意一次遍历,只会处理根结点,新加入的子结点不会被遍历到

        list<ExtractorNode>::iterator ending = lNodes.end();

        while (lit != lNodes.end()) {

            if (lit->bNoMore) {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            } else {
                // If more than one point, subdivide
                // 构造四个子结点,并将当前特征点分到这四个子结点中
                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);  //当前节点分成四个象限

                // Add childs if they contain points
                if (n1.vKeys.size() > 0) {
                    // 注意这里是从前部加进去的
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1) {
                        nToExpand++;
                        // 若子结点中特征点超过1,则将其指针记录在vSizeAndPointerToNode之中
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));  //记录下当前所分节点的特征点数目
                        // 当前 lNodes.front() = n1                                                     //以及当前所分节点的地址
                        // n1.lit = lNode.begin(),也即它自己的迭代器指向它在lNode中的位置
                        lNodes.front().lit = lNodes.begin();  //记录下当前所分节点的上一节点的地址
                    }
                }
                if (n2.vKeys.size() > 0) {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0) {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0) {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                // 分完过后,删除当前根结点
                lit = lNodes.erase(lit);
                continue;
            }

        }

        cout << "prevSize"<<prevSize << endl;
        cout << "nToExpand"<<nToExpand << endl;



        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int) lNodes.size() >= N || (int) lNodes.size() == prevSize) {
            bFinish = true;
        } else if (((int) lNodes.size() + nToExpand * 3) > N) {
            // 对于头一次从根结点,假设根结点只有1个,那么分出来的子结点只有4个,那么下面的的条件是进入不了的
            // 当分的差不多时,只有个别结点未分时,进入以下逻辑
            while (!bFinish) {

                prevSize = lNodes.size();

                vector<pair<int, ExtractorNode *> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                // 将元素按升序排列,这里的假设是包含的特征点少的结点往往已经不可分了
                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
                    ExtractorNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0) {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0) {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0) {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0) {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int) lNodes.size() >= N)
                        break;
                }

                if ((int) lNodes.size() >= N || (int) lNodes.size() == prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    Mat image_dev = OctTree_image.rowRange(minY, maxY).colRange(minX, maxX);
    for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++) {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint *pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        // 如果vNodeKeys中只有1个关键点,则直接将其加入到最终结果点中
        // 如果最终的结点中有多个点,那么只有最大相应值的结点会被留下了
        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }


        lit->DrawGrid(image_dev);
        cv::drawKeypoints(image_dev, vResultKeys, image_dev, Scalar(255, 0, 255));
        imshow("i2", image_dev);
        waitKey(1);


        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}



void computeOrientation(const Mat &image, vector<KeyPoint> &keypoints, const vector<int> &umax) {
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint) {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);

    }
}

float IC_Angle(const Mat &image, Point2f pt, const vector<int> &u_max) -{
    int m_01 = 0, m_10 = 0;
// 灰度质心法计算特征点方向


    const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));    // 得到中心位置的指针

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)	// 对 v=0 这一行单独计算
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int) image.step1();  // 这边要注意图像的step不一定是图像的宽度; step用于要访问某区域下一行时候，数据指针的步长
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float) m_01, (float) m_10);
}
