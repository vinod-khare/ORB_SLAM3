#include "Frame.h"
#include "KannalaBrandt8.h"
#include "MapPoint.h"
#include "Pinhole.h"
#include "keypoint_extractor.h"

#include <algorithm>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEST_CASE("Optical flow reuses frame pyramids and matches raw image tracking")
{
    cv::Mat previous_image(128, 128, CV_8UC1);
    cv::RNG random(42);
    random.fill(previous_image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(previous_image, previous_image, cv::Size(5, 5), 0);

    cv::Mat       current_image;
    const cv::Mat transform = cv::Mat(cv::Matx23d(1, 0, 2, 0, 1, 1));
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

TEST_CASE("Flow frames defer extraction and refresh current-image features without replacing state")
{
    const int  camera_kind = GENERATE(0, 1, 2, 3); // monocular, rectified stereo, fisheye stereo, distorted monocular
    const bool stereo      = camera_kind == 1 || camera_kind == 2;

    cv::Mat    image(256, 256, CV_8UC1);
    cv::RNG    random(17);

    random.fill(image, cv::RNG::UNIFORM, 0, 256);

    ORB_SLAM3::keypoint_extractor left_extractor(400, 1.2f, 4, 20, 7);
    ORB_SLAM3::keypoint_extractor right_extractor(400, 1.2f, 4, 20, 7);
    ORB_SLAM3::Pinhole            pinhole({200, 200, 128, 128});
    ORB_SLAM3::KannalaBrandt8     left_camera({200, 200, 128, 128, 0, 0, 0, 0});
    ORB_SLAM3::KannalaBrandt8     right_camera({200, 200, 128, 128, 0, 0, 0, 0});

    left_camera.mvLappingArea  = {0, 256};
    right_camera.mvLappingArea = {0, 256};

    auto    calibration        = pinhole.toK();
    cv::Mat distortion         = cv::Mat::zeros(4, 1, CV_32F);

    if (camera_kind == 3)
    {
        distortion.at<float>(0) = -0.1f;
    }
    Sophus::SE3f relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0, 0));
    ORB_SLAM3::Frame::mbInitialComputations = true;
    ORB_SLAM3::Frame previous;
    if (!stereo)
    {
        previous = ORB_SLAM3::Frame(image, 1.0, &left_extractor, nullptr, &pinhole, distortion, 20, 40);
    }
    else if (camera_kind == 1)
    {
        previous = ORB_SLAM3::Frame(image, image, 1.0, &left_extractor, &right_extractor, nullptr, calibration, distortion, 20, 40, &pinhole);
    }
    else
    {
        previous = ORB_SLAM3::Frame(image, image, 1.0, &left_extractor, &right_extractor, nullptr, calibration, distortion, 20, 40, &left_camera, &right_camera, relative_pose);
    }
    previous.mpLastKeyFrame = nullptr;
    previous.mnDataset      = 0;
    REQUIRE(previous.N > 0);
    REQUIRE(previous.display_keys.size() == previous.mvKeys.size());
    REQUIRE(previous.display_keys_right.size() == previous.mvKeysRight.size());
    REQUIRE(previous.display_keys.front().size == 7 * previous.mvScaleFactors[previous.mvKeys.front().octave]);
    REQUIRE(previous.display_keys.front().angle == -1);
    const ORB_SLAM3::Frame copied(previous);
    REQUIRE(copied.display_keys.front().size == previous.display_keys.front().size);
    REQUIRE(copied.display_keys_right.size() == previous.display_keys_right.size());
    REQUIRE_FALSE(previous.mDescriptors.empty());
    previous.get_optical_flow_pyramid(image);
    const auto       next_id = ORB_SLAM3::Frame::nNextId;
    ORB_SLAM3::Frame current(previous, 1.05);
    REQUIRE(current.mnId == next_id);
    REQUIRE(ORB_SLAM3::Frame::nNextId == next_id + 1);
    REQUIRE(current.flow_only);
    REQUIRE(current.N == 0);
    REQUIRE(current.display_keys.empty());
    REQUIRE(current.display_keys_right.empty());
    REQUIRE(current.mDescriptors.empty());
    REQUIRE(current.mDescriptorsRight.empty());
    REQUIRE(current.optical_flow_pyramid.empty());
    REQUIRE(current.mpPrevFrame == &previous);
    REQUIRE_FALSE(current.HasPose());
    REQUIRE_FALSE(current.imuIsPreintegrated());

    ORB_SLAM3::MapPoint point;
    cv::KeyPoint        tracked(73.25f, 81.75f, 31.0f);
    const cv::KeyPoint  native_tracked(tracked.pt, 7.f, -1.f);
    current.set_tracked_features({tracked}, {&point}, {native_tracked});
    REQUIRE(current.N == 1);
    REQUIRE(current.mvKeys.front().pt == tracked.pt);
    REQUIRE(current.display_keys.front().pt == tracked.pt);
    REQUIRE(current.display_keys.front().size == 7.f);
    REQUIRE(current.display_keys.front().angle == -1.f);
    REQUIRE(current.mvKeys.front().size == 31.f);
    REQUIRE(current.display_keys_right.empty());
    std::vector<cv::Point2f> expected_undistorted;
    cv::undistortPoints(std::vector<cv::Point2f>{tracked.pt}, expected_undistorted, calibration, distortion, cv::noArray(), calibration);
    REQUIRE(cv::norm(current.mvKeysUn.front().pt - expected_undistorted.front()) < 0.001);
    REQUIRE(current.mvpMapPoints.front() == &point);
    REQUIRE(current.mvuRight.front() == -1);
    REQUIRE(current.mvDepth.front() == -1);
    REQUIRE(current.mDescriptors.empty());
    REQUIRE(current.Nleft == (camera_kind == 2 ? 1 : -1));
    REQUIRE(current.mvKeysRight.empty());
    if (camera_kind == 2)
    {
        REQUIRE(current.Nright == 0);
        REQUIRE(current.mvLeftToRightMatch == std::vector<int>{-1});
    }

    const auto pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f(1, 2, 3));
    current.SetPose(pose);
    current.SetVelocity(Eigen::Vector3f(4, 5, 6));
    ORB_SLAM3::IMU::Preintegrated frame_integration;
    ORB_SLAM3::IMU::Preintegrated keyframe_integration;
    current.mpImuPreintegratedFrame = &frame_integration;
    current.mpImuPreintegrated      = &keyframe_integration;
    current.setIntegrated();
    current.get_optical_flow_pyramid(image);
    const auto *pyramid_data = current.optical_flow_pyramid.front().data;
    current.extract_features(image, camera_kind == 2 ? static_cast<ORB_SLAM3::GeometricCamera *>(&left_camera) : &pinhole, stereo ? image : cv::Mat(),
                             camera_kind == 2 ? &right_camera : nullptr);
    REQUIRE_FALSE(current.flow_only);
    REQUIRE(current.mnId == next_id);
    REQUIRE(ORB_SLAM3::Frame::nNextId == next_id + 1);
    REQUIRE(current.mTimeStamp == 1.05);
    REQUIRE(current.mpPrevFrame == &previous);
    REQUIRE(current.imuIsPreintegrated());
    REQUIRE(current.mpImuPreintegratedFrame == &frame_integration);
    REQUIRE(current.mpImuPreintegrated == &keyframe_integration);
    REQUIRE(current.GetPose().matrix().isApprox(pose.matrix()));
    REQUIRE(current.GetVelocity().isApprox(Eigen::Vector3f(4, 5, 6)));
    REQUIRE(current.optical_flow_pyramid.front().data == pyramid_data);
    REQUIRE(current.N == previous.N);
    REQUIRE(current.mDescriptors.rows == current.N);
    REQUIRE(cv::norm(current.mDescriptors, previous.mDescriptors, cv::NORM_INF) == 0);
    REQUIRE(current.mvpMapPoints.size() == static_cast<std::size_t>(current.N));
    REQUIRE(std::all_of(current.mvpMapPoints.begin(), current.mvpMapPoints.end(), [](const auto *entry) { return entry == nullptr; }));
    REQUIRE(current.mvKeys.size() == previous.mvKeys.size());
    REQUIRE(current.mvKeysRight.size() == previous.mvKeysRight.size());
    REQUIRE(current.display_keys.size() == previous.display_keys.size());
    REQUIRE(current.display_keys_right.size() == previous.display_keys_right.size());
    REQUIRE(current.display_keys.front().size == previous.display_keys.front().size);
    REQUIRE(current.display_keys.front().angle == previous.display_keys.front().angle);
    REQUIRE(current.mvLeftToRightMatch == previous.mvLeftToRightMatch);
    REQUIRE(current.mvDepth == previous.mvDepth);
    REQUIRE(current.mBowVec.empty());
    REQUIRE(current.mFeatVec.empty());
    const auto first_grid = current.GetFeaturesInArea(128, 128, 256);
    current.extract_features(image, camera_kind == 2 ? static_cast<ORB_SLAM3::GeometricCamera *>(&left_camera) : &pinhole, stereo ? image : cv::Mat(),
                             camera_kind == 2 ? &right_camera : nullptr);
    REQUIRE(current.GetFeaturesInArea(128, 128, 256) == first_grid);

    if (camera_kind == 2)
    {
        const cv::Mat blank = cv::Mat::zeros(image.size(), image.type());
        current.extract_features(blank, &left_camera, blank, &right_camera);
        REQUIRE(current.N == 0);
        REQUIRE(current.Nleft == 0);
        REQUIRE(current.Nright == 0);
        current.set_tracked_features({tracked}, {&point});
        REQUIRE(current.Nleft == 1);
        REQUIRE(current.Nright == 0);
        REQUIRE(current.mvKeysUn.front().pt == tracked.pt);
        REQUIRE(current.mvLeftToRightMatch == std::vector<int>{-1});
    }
}

TEST_CASE("RGB-D frames undistort tracked features without a camera")
{
    cv::Mat image(256, 256, CV_8UC1);
    cv::RNG random(17);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 4, 20, 7);
    cv::Mat                       calibration = cv::Mat(cv::Matx33f(200, 0, 128, 0, 200, 128, 0, 0, 1));
    cv::Mat                       distortion  = cv::Mat(cv::Vec4f(-0.1f, 0, 0, 0));
    const cv::Mat                 depth(image.size(), CV_32F, cv::Scalar(2));
    ORB_SLAM3::Frame::mbInitialComputations = true;
    ORB_SLAM3::Frame frame(image, depth, 1.0, &extractor, nullptr, calibration, distortion, 20, 40);
    REQUIRE(frame.N > 0);
    REQUIRE(frame.Nleft == -1);
    REQUIRE(std::all_of(frame.mvDepth.begin(), frame.mvDepth.end(), [](float value) { return value == 2.0f; }));
    REQUIRE(std::all_of(frame.mvuRight.begin(), frame.mvuRight.end(), [](float value) { return value >= 0; }));

    ORB_SLAM3::MapPoint point;
    const cv::KeyPoint  tracked(73.25f, 81.75f, 31.0f);
    frame.set_tracked_features({tracked}, {&point});
    std::vector<cv::Point2f> expected;
    cv::undistortPoints(std::vector<cv::Point2f>{tracked.pt}, expected, calibration, distortion, cv::noArray(), calibration);
    REQUIRE(cv::norm(frame.mvKeysUn.front().pt - expected.front()) < 0.001);
    REQUIRE(frame.Nleft == -1);
}

TEST_CASE("Flow quality requires geometric inliers and spatial coverage")
{
    ORB_SLAM3::Frame frame;
    frame.mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::MapPoint       point;
    std::vector<cv::KeyPoint> keys;
    for (int i = 0; i < 100; ++i)
    {
        keys.emplace_back(static_cast<float>(16 + (i % 4) * 32), static_cast<float>(16 + ((i / 4) % 4) * 32), 31.0f);
    }
    frame.set_tracked_features(keys, std::vector<ORB_SLAM3::MapPoint *>(100, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    std::fill(frame.mvbOutlier.begin(), frame.mvbOutlier.begin() + 40, true);
    REQUIRE(frame.reliable_flow_inliers({128, 128})); // exactly 60%
    frame.mvbOutlier[40] = true;
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<cv::KeyPoint>(100, cv::KeyPoint(16, 16, 31)), std::vector<ORB_SLAM3::MapPoint *>(100, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<cv::KeyPoint>(keys.begin(), keys.begin() + 49), std::vector<ORB_SLAM3::MapPoint *>(49, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<cv::KeyPoint>(keys.begin(), keys.begin() + 50), std::vector<ORB_SLAM3::MapPoint *>(50, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    std::vector<cv::KeyPoint> six_cells;
    for (int i = 0; i < 60; ++i)
    {
        six_cells.emplace_back(static_cast<float>(16 + (i % 3) * 32), static_cast<float>(16 + ((i / 3) % 2) * 32), 31.0f);
    }
    frame.set_tracked_features(six_cells, std::vector<ORB_SLAM3::MapPoint *>(60, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    for (auto &key : six_cells)
    {
        if (key.pt == cv::Point2f(80, 48))
        {
            key.pt = cv::Point2f(16, 16);
        }
    }
    frame.set_tracked_features(six_cells, std::vector<ORB_SLAM3::MapPoint *>(60, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(keys, std::vector<ORB_SLAM3::MapPoint *>(100, nullptr));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    for (auto &key : keys)
    {
        key.pt.x = std::numeric_limits<float>::quiet_NaN();
    }
    frame.mvKeys = keys;
    frame.mvpMapPoints.assign(100, &point);
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    REQUIRE_FALSE(frame.reliable_flow_inliers({0, 128}));
}