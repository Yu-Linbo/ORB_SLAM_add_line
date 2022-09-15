//
// Created by zlkj on 2022/8/15.
//

#ifndef ORB_SLAM2_LKODOMETER_H
#define ORB_SLAM2_LKODOMETER_H

#include "Frame.h"
#include "KeyFrame.h"
#include <opencv2/line_descriptor/descriptor.hpp>

namespace ORB_SLAM2 {

    class LKOdometer {
        public:
        LKOdometer(cv::Mat &first);

        void odometer(Frame &lastF, Frame &curF , cv::Mat &curImage);

        cv::Mat mLastImage;
        cv::Mat mCurImage;
    };
}

#endif //ORB_SLAM2_LKODOMETER_H
