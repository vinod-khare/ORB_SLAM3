#include "Frame.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Optical flow reuses frame pyramids and matches raw image tracking")
{
    cv::Mat previous_image(128, 128, CV_8UC1);
    cv::RNG random(42);
    random.fill(previous_image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(previous_image, previous_image, cv::Size(5, 5), 0);

    cv::Mat       current_image;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, 2, 0, 1, 1);
    cv::warpAffine(previous_image, current_image, transform, previous_image.size());

    ORB_SLAM3::Frame previous_frame;
    ORB_SLAM3::Frame current_frame;
    const auto      &previous_pyramid = previous_frame.get_optical_flow_pyramid(previous_image);
    const auto      &current_pyramid  = current_frame.get_optical_flow_pyramid(current_image);

    REQUIRE(previous_pyramid.size() == 6);
    REQUIRE(previous_frame.get_optical_flow_pyramid(previous_image).front().data == previous_pyramid.front().data);
    REQUIRE(previous_pyramid.front().data != previous_image.data);
    REQUIRE(cv::norm(previous_pyramid.front(), previous_image, cv::NORM_INF) == 0);

    const std::vector<cv::Point2f> previous_points{
        {32, 32},
        {64, 32},
        {32, 64},
        {64, 64},
        {96, 96}
    };
    std::vector<cv::Point2f>   raw_points;
    std::vector<cv::Point2f>   cached_points;
    std::vector<unsigned char> raw_status;
    std::vector<unsigned char> cached_status;
    std::vector<float>         raw_error;
    std::vector<float>         cached_error;

    cv::calcOpticalFlowPyrLK(previous_image, current_image, previous_points, raw_points, raw_status, raw_error);
    cv::calcOpticalFlowPyrLK(previous_pyramid, current_pyramid, previous_points, cached_points, cached_status, cached_error);
    REQUIRE(cached_status == raw_status);
    for (std::size_t index = 0; index < previous_points.size(); ++index)
    {
        REQUIRE(raw_status[index] == 1);
        REQUIRE(cv::norm(cached_points[index] - raw_points[index]) < 0.001);
    }

    std::vector<cv::Point2f> raw_backward_points;
    std::vector<cv::Point2f> cached_backward_points;
    cv::calcOpticalFlowPyrLK(current_image, previous_image, raw_points, raw_backward_points, raw_status, raw_error);
    cv::calcOpticalFlowPyrLK(current_pyramid, previous_pyramid, cached_points, cached_backward_points, cached_status, cached_error);
    REQUIRE(cached_status == raw_status);
    REQUIRE(std::count(raw_status.begin(), raw_status.end(), 1) > 0);
    for (std::size_t index = 0; index < previous_points.size(); ++index)
    {
        if (raw_status[index] != 0)
        {
            REQUIRE(cv::norm(cached_backward_points[index] - raw_backward_points[index]) < 0.001);
        }
    }

    previous_image.setTo(0);
    REQUIRE(cv::norm(previous_pyramid.front(), cv::NORM_INF) > 0);
}