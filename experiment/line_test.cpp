//
// Created by ylb on 2022/7/7.
//
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <algorithm>

#include <string>
#include<opencv2/opencv.hpp>
#include "include/ExtractLineSegment.h"
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat _preImage=imread("1.png");
    Mat _currentImage=imread("2.png");
    Mat outImg;
    ORB_SLAM2::LineSegment extractor;

    vector<KeyLine> keylines1,keylines2;
    Mat ldesc1, ldesc2;
    vector<Vector3d> keylineFunctions1, keylineFunctions2;

    extractor.ExtractLineSegment(_preImage, keylines1, ldesc1, keylineFunctions1);
    extractor.ExtractLineSegment(_currentImage, keylines2, ldesc2, keylineFunctions2);
    extractor.LineSegmentMatch(ldesc1,ldesc2);

    cout<< "startPoint: " <<  keylines1[0].startPointX << " " << keylines1[0].startPointY <<endl;
    cout<< "endPoint: " <<  keylines1[0].endPointX << " " << keylines1[0].endPointY <<endl;
    // line equation
    float i= min(keylines1[0].startPointX,keylines1[0].endPointX);
    float max_x= max(keylines1[0].startPointX,keylines1[0].endPointX);
    float fa= (keylines1[0].startPointY-keylines1[0].endPointY)/(keylines1[0].startPointX-keylines1[0].endPointX);
    float fb= keylines1[0].startPointY - fa*keylines1[0].startPointX;
    // 记录深度梯度
//    for(;i<max_x;i++){
//        // cout<< "x=" << i << " y="<< i*fa + fb<< endl;
//        _preImage
//    }
    //sort(extractor.LineSegmentMatch.begin(), extractor.LineSegmentMatch.end(), sort_descriptor_by_queryIdx());

    std::vector<char> mask( extractor.line_matches.size(), 1 );
    drawLineMatches( _preImage, keylines1, _currentImage, keylines2, extractor.line_matches[0], outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask,DrawLinesMatchesFlags::DEFAULT );
    imshow( "Matches", outImg );
    // Wait for a keystroke in the window
    waitKey(0);
    return 0;
}