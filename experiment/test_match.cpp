//
// Created by ylb on 2022/12/28.
//

#include <iostream>
#include <cv.hpp>
#include <chrono>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include "include/ORBextractor.h"
#include "include/ExtractLineSegment.h"

using namespace cv;

void reduceVector(std::vector<cv::Point2f> v, std::vector<uchar> status, std::vector<cv::Point2f> &v1);
void reduceVector_line(std::vector<cv::Point2f> v1, std::vector<cv::Point2f> v2, std::vector<uchar> status1, std::vector<uchar> status2, std::vector<cv::Point2f> &match_v1, std::vector<cv::Point2f> &match_v2);

void drawMatches(const std::vector<cv::Point2f> &points_1, const std::vector<cv::Point2f> &points_2,
                 const cv::Mat &image_src_,
                 const cv::Mat &image_dst_, cv::Mat &out_image_, int circle_radius_);

void drawMatches_line(const std::vector<cv::Point2f> &points_1, const std::vector<cv::Point2f> &points_2,
                      const std::vector<cv::Point2f> &points_3, const std::vector<cv::Point2f> &points_4,
                      const cv::Mat &image_src_,
                      const cv::Mat &image_dst_, cv::Mat &out_image_, int line_radius_);

void timeCost(const std::chrono::time_point<std::chrono::system_clock>,
              const std::chrono::time_point<std::chrono::system_clock>,
              const std::chrono::time_point<std::chrono::system_clock>, std::string);

int main() {
    std::chrono::time_point<std::chrono::system_clock> start, end1, end2;
    start = std::chrono::system_clock::now();
    std::string first_img_path = "/home/ylb/my_paper_code/ORB_SLAM_add_line/experiment/1.png", second_img_path = "/home/ylb/my_paper_code/ORB_SLAM_add_line/experiment/2.png";
    cv::Mat img1, img2, img1_, img2_, match, ransac;
    int MAX_CNT = 500;
    double MIN_DIST = 10;
    std::vector<cv::Point2f> n_pts_1, n_pts_2, match_pts_1, match_pts_2, ran_pts_1, ran_pts_2;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<uchar> status;
    std::vector<float> err;

    // opencv 提供的orb提取器
    // Ptr<ORB> detector = ORB::create();

    //orbslam2 中的orb提取器
    ORB_SLAM2::ORBextractor* mpORBextractorLeft = new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7);

    /// line 特征提取
    vector<KeyLine> keylines1,keylines2;
    Mat ldesc1; //line特征描述子
    vector<Vector3d> keylineFunctions1; //line特征系数
    ORB_SLAM2::LineSegment extractor; // line 特征提取器

    end1 = std::chrono::system_clock::now();
    timeCost(start, start, end1, "prepare");

    img1 = cv::imread(first_img_path, cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(second_img_path, cv::IMREAD_GRAYSCALE);
    img1_ = cv::imread(first_img_path);
    img2_ = cv::imread(second_img_path);
    end2 = std::chrono::system_clock::now();
    timeCost(start, end1, end2, "read image");

    // 提取特征点

    (*mpORBextractorLeft)(img1,cv::Mat(),keypoints_1);
    (*mpORBextractorLeft)(img2,cv::Mat(),keypoints_2);

    // 画特征点
    cv::Mat outimage1;
    drawKeypoints(img1_,keypoints_1,outimage1);
    cv::imshow("image1特征点",outimage1);
    cv::waitKey(0);
//    KeyPoint::convert(keypoints_1,n_pts_1);
//    ORBmatcher matcher(0.7,true);
//    vector<MapPoint*> vpMapPointMatches;

    return 0;
}

void drawMatches(const std::vector<cv::Point2f> &points_1, const std::vector<cv::Point2f> &points_2,
                 const cv::Mat &image_src_,
                 const cv::Mat &image_dst_, cv::Mat &out_image_, int circle_radius_) {
    // Final image
    out_image_.create(image_src_.rows, // Height
                      2 * image_src_.cols, // Width
                      image_src_.type()); // Type

    cv::Mat roi_img_result_left =
            out_image_(cv::Rect(0, 0, image_src_.cols, image_src_.rows)); // Img1 will be on the left part
    cv::Mat roi_img_result_right =
            out_image_(cv::Rect(image_src_.cols, 0, image_dst_.cols,
                                image_dst_.rows)); // Img2 will be on the right part, we shift the roi of img1.cols on the right

    cv::Mat roi_image_src = image_src_(cv::Rect(0, 0, image_src_.cols, image_src_.rows));
    cv::Mat roi_image_dst = image_dst_(cv::Rect(0, 0, image_dst_.cols, image_dst_.rows));

    roi_image_src.copyTo(roi_img_result_left); //Img1 will be on the left of imgResult
    roi_image_dst.copyTo(roi_img_result_right); //Img2 will be on the right of imgResult

    for (int i = 0; i < points_1.size(); ++i) {
        cv::Point2d pt1(points_1.at(i).x,
                        points_1.at(i).y);
        cv::Point2d pt2(image_dst_.cols + points_2.at(i).x,
                        points_2.at(i).y);

        cv::Scalar color(255 * static_cast<double>(rand()) / RAND_MAX,
                         255 * static_cast<double>(rand()) / RAND_MAX,
                         255 * static_cast<double>(rand()) / RAND_MAX);

        cv::circle(out_image_, pt1, circle_radius_, color, static_cast<int>(circle_radius_ * 0.4));
        cv::circle(out_image_, pt2, circle_radius_, color, static_cast<int>(circle_radius_ * 0.4));
        cv::line(out_image_, pt1, pt2, color, 2);
    }
}

void drawMatches_line(const std::vector<cv::Point2f> &points_1, const std::vector<cv::Point2f> &points_2, const std::vector<cv::Point2f> &points_3, const std::vector<cv::Point2f> &points_4,
                      const cv::Mat &image_src_,
                      const cv::Mat &image_dst_, cv::Mat &out_image_, int line_radius_) {
    // Final image
    out_image_.create(image_src_.rows, // Height
                      2 * image_src_.cols, // Width
                      image_src_.type()); // Type

    cv::Mat roi_img_result_left =
            out_image_(cv::Rect(0, 0, image_src_.cols, image_src_.rows)); // Img1 will be on the left part
    cv::Mat roi_img_result_right =
            out_image_(cv::Rect(image_src_.cols, 0, image_dst_.cols,
                                image_dst_.rows)); // Img2 will be on the right part, we shift the roi of img1.cols on the right

    cv::Mat roi_image_src = image_src_(cv::Rect(0, 0, image_src_.cols, image_src_.rows));
    cv::Mat roi_image_dst = image_dst_(cv::Rect(0, 0, image_dst_.cols, image_dst_.rows));

    roi_image_src.copyTo(roi_img_result_left); //Img1 will be on the left of imgResult
    roi_image_dst.copyTo(roi_img_result_right); //Img2 will be on the right of imgResult

    for (int i = 0; i < points_1.size(); ++i) {
        cv::Point2d pt1(points_1.at(i).x,
                        points_1.at(i).y);
        cv::Point2d pt2(points_2.at(i).x,
                        points_2.at(i).y);
        cv::Point2d pt3(image_dst_.cols + points_3.at(i).x,
                        points_3.at(i).y);
        cv::Point2d pt4(image_dst_.cols + points_4.at(i).x,
                        points_4.at(i).y);

        cv::Scalar color(255 * static_cast<double>(rand()) / RAND_MAX,
                         255 * static_cast<double>(rand()) / RAND_MAX,
                         255 * static_cast<double>(rand()) / RAND_MAX);

        cv::line(out_image_, pt1, pt2, color, 1.5*line_radius_, 8, 0);
        cv::line(out_image_, pt3, pt4, color, 1.5*line_radius_, 8, 0);

        cv::line(out_image_, pt2, pt4, color, 0.5*line_radius_, 8, 0);
    }
}

void timeCost(const std::chrono::time_point<std::chrono::system_clock> start,
              const std::chrono::time_point<std::chrono::system_clock> end1,
              const std::chrono::time_point<std::chrono::system_clock> end2, std::string name) {
    std::chrono::duration<double> elapsed_seconds;
    elapsed_seconds = end2 - end1;
    printf("%s %f secs\n", name.c_str(), elapsed_seconds.count());
    elapsed_seconds = end2 - start;
    printf("累计用时 = %f secs\n", elapsed_seconds.count());
}

void reduceVector(std::vector<cv::Point2f> v, std::vector<uchar> status, std::vector<cv::Point2f> &v1) {
    for (int i = 0; i < int(v.size()); i++) {
        if (status[i]) {
            v1.push_back(v.at(i));
        }
    }
}

void reduceVector_line(std::vector<cv::Point2f> v1, std::vector<cv::Point2f> v2, std::vector<uchar> status1, std::vector<uchar> status2, std::vector<cv::Point2f> &match_v1, std::vector<cv::Point2f> &match_v2) {
    for (int i = 0; i < int(v1.size()); i++) {
        if (status1[i] && status2[i]) {
            match_v1.push_back(v1.at(i));
            match_v2.push_back(v2.at(i));
        }
    }
}