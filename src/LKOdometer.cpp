//
// Created by zlkj on 2022/8/15.
//

#include "LKOdometer.h"
#include <iostream>
#include <fstream>  //进行文件io
#include <list>
#include <vector>
#include <chrono>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

//namespace
using namespace std;
using namespace cv::line_descriptor;
using namespace Eigen;
using namespace cv;

namespace ORB_SLAM2
{

    LKOdometer::LKOdometer(cv::Mat &firstImage){
        mLastImage = firstImage;
        mCurImage = firstImage;
    }

    void LKOdometer::odometer(Frame &lastF, Frame &curF , cv::Mat &curImage)
    {
        // 获取当前帧
        mCurImage = curImage;

        // 获取上一帧特征点与特征线
        vector<cv::KeyPoint> kps=lastF.mvKeysUn;
        vector<KeyLine> kls=lastF.mvKeylinesUn;

            //对其他帧用LK跟踪特征点
            vector<cv::Point2f> cur_key_points;
            vector<cv::Point2f> prev_key_points;

            for(auto kp:kps){
                prev_key_points.push_back(kp.pt);
            }
            vector<unsigned char> status;
            vector<float> error;

            chrono::steady_clock::time_point t1=chrono::steady_clock::now();

            cv::calcOpticalFlowPyrLK(mLastImage, mCurImage, prev_key_points, cur_key_points, status, error);
            //status: 输出状态向量，如果找到了对应特征的流，则将向量的相应元素设置为1；否则，置0
            //error: 误差输出向量，vector的每个元素被设置为对应特征的误差

            chrono::steady_clock::time_point t2=chrono::steady_clock::now();
            //计算跟踪一次需要的时间
            chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double >>(t2-t1);
            cout<<"LK FLOW use time: "<<time_used.count()<<" seconds."<<endl;

            //把跟丢的点去掉
            int i=0;
            for(auto iter=cur_key_points.begin(); iter!=cur_key_points.end(); iter++){
                // 若跟踪成功
                if(status[i]==1){
                    curF.mvKeysUn.push_back(KeyPoint(cur_key_points[i],1.f));
                    continue;
                }
                i++;
            }
            cout<<"tracked keypoints: "<<curF.mvKeysUn.size()<<endl;

            if(curF.mvKeysUn.size()==0){
                cout<<"all keypoints are lost. "<<endl;
            }

        // 当前帧设为上一帧
        mLastImage = curImage;
    }
}
