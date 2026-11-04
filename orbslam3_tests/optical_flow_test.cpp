#include "Frame.h"
#include "KannalaBrandt8.h"
#include "MapPoint.h"
#include "Pinhole.h"
#include "Tracking.h"
#include "keypoint_extractor.h"

#include <algorithm>
#include <limits>
#include <memory>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEST_CASE("Default and copied frames own a valid IMU integration lock")
{
    ORB_SLAM3::Frame frame;
    REQUIRE_FALSE(frame.imuIsPreintegrated());

    frame.setIntegrated();
    REQUIRE(frame.imuIsPreintegrated());

    frame.mK = cv::Mat::eye(3, 3, CV_32F);
    const ORB_SLAM3::Frame copied(frame);
    REQUIRE(copied.imuIsPreintegrated());
}

TEST_CASE("Frame construction initializes state without extraction")
{
    const cv::Mat    calibration = cv::Mat(cv::Matx33f(200, 0, 128, 0, 200, 128, 0, 0, 1));
    const cv::Mat    distortion  = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::Frame previous(1.0, nullptr, calibration, distortion, 20, 40);
    previous.SetVelocity(Eigen::Vector3f(1, 2, 3));
    const Sophus::SE3f relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0, 0));
    const auto         next_id = ORB_SLAM3::Frame::nNextId;
    ORB_SLAM3::Frame   current(1.05, nullptr, calibration, distortion, 20, 40, &previous, {}, relative_pose);
    REQUIRE(current.mnId == next_id);
    REQUIRE(ORB_SLAM3::Frame::nNextId == next_id + 1);
    REQUIRE(current.mTimeStamp == 1.05);
    REQUIRE(current.N == 0);
    REQUIRE(current.mvKeys.empty());
    REQUIRE(current.mvKeysRight.empty());
    REQUIRE(current.mvScaleFactors.empty());
    REQUIRE(current.optical_flow_pyramid.empty());
    REQUIRE(current.mpPrevFrame == &previous);
    REQUIRE(current.HasVelocity());
    REQUIRE(current.GetVelocity().isApprox(previous.GetVelocity()));
    REQUIRE(current.GetRelativePoseTlr().matrix().isApprox(relative_pose.matrix()));
    REQUIRE_FALSE(current.HasPose());
    REQUIRE_FALSE(current.imuIsPreintegrated());
    REQUIRE(cv::norm(current.mK, calibration, cv::NORM_INF) == 0);
    REQUIRE(current.mK.data != calibration.data);
}

TEST_CASE("Frame feature data survives extractor reuse and destruction")
{
    const auto descriptor_type = GENERATE(ORB_SLAM3::keypoint_descriptor_type::orb, ORB_SLAM3::keypoint_descriptor_type::brisk, ORB_SLAM3::keypoint_descriptor_type::sift);
    cv::Mat    image(256, 256, CV_8UC1);
    cv::RNG    random(17);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{200, 200, 128, 128}));
    const cv::Mat    distortion = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::Frame frame(1.0, nullptr, cameras[0]->toK(), distortion, 20, 40);
    ORB_SLAM3::Frame::mbInitialComputations = true;
    cv::Mat descriptors;
    cv::Mat place_descriptors;
    {
        ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 4, 20, 7, ORB_SLAM3::keypoint_detector_type::fast, descriptor_type);
        ORB_SLAM3::Tracking::extract_features(frame, extractor, image, cameras);
        REQUIRE(frame.N > 0);
        REQUIRE(frame.mnScaleLevels == extractor.get_levels());
        REQUIRE(frame.mfScaleFactor == extractor.get_scale_factor());
        REQUIRE(frame.mvScaleFactors == extractor.get_scale_factors());
        REQUIRE(frame.mvInvScaleFactors == extractor.get_inverse_scale_factors());
        REQUIRE(frame.mvLevelSigma2 == extractor.get_scale_sigma_squares());
        REQUIRE(frame.mvInvLevelSigma2 == extractor.get_inverse_scale_sigma_squares());
        descriptors = descriptor_matrix(frame.mvKeys).clone();
        cv::vconcat(place_descriptor_vector(frame.mvKeys), place_descriptors);
        ORB_SLAM3::Frame other(2.0, nullptr, cameras[0]->toK(), distortion, 20, 40);
        ORB_SLAM3::Tracking::extract_features(other, extractor, cv::Mat::zeros(image.size(), image.type()), cameras);
        REQUIRE(other.N == 0);
    }
    REQUIRE(cv::norm(descriptor_matrix(frame.mvKeys), descriptors, cv::NORM_INF) == 0);
    cv::Mat retained_place_descriptors;
    cv::vconcat(place_descriptor_vector(frame.mvKeys), retained_place_descriptors);
    REQUIRE(cv::norm(retained_place_descriptors, place_descriptors, cv::NORM_INF) == 0);
    const ORB_SLAM3::Frame copied(frame);
    REQUIRE(copied.N == frame.N);
    REQUIRE(cv::norm(descriptor_matrix(copied.mvKeys), descriptors, cv::NORM_INF) == 0);
}

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

    cv::Mat image(256, 256, CV_8UC1);
    cv::RNG random(17);

    random.fill(image, cv::RNG::UNIFORM, 0, 256);

    ORB_SLAM3::keypoint_extractor left_extractor(400, 1.2f, 4, 20, 7);
    ORB_SLAM3::keypoint_extractor right_extractor(400, 1.2f, 4, 20, 7);
    ORB_SLAM3::camera_vector      cameras;
    for (int camera_index = 0; camera_index < (stereo ? 2 : 1); ++camera_index)
    {
        if (camera_kind == 2)
        {
            auto camera           = std::make_unique<ORB_SLAM3::KannalaBrandt8>(std::vector<float>{200, 200, 128, 128, 0, 0, 0, 0});
            camera->mvLappingArea = {0, 256};
            cameras.push_back(std::move(camera));
        }
        else
        {
            cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{200, 200, 128, 128}));
        }
    }

    auto    calibration = cameras[0]->toK();
    cv::Mat distortion  = cv::Mat::zeros(4, 1, CV_32F);

    if (camera_kind == 3)
    {
        distortion.at<float>(0) = -0.1f;
    }
    Sophus::SE3f relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0, 0));
    ORB_SLAM3::Frame::mbInitialComputations = true;
    ORB_SLAM3::Frame previous(1.0, nullptr, calibration, distortion, 20, 40, nullptr, {}, relative_pose);
    ORB_SLAM3::Tracking::extract_features(previous, left_extractor, image, cameras, stereo ? image : cv::Mat(), stereo ? &right_extractor : nullptr);
    previous.mpLastKeyFrame = nullptr;
    previous.mnDataset      = 0;
    REQUIRE(previous.N > 0);
    REQUIRE(previous.mvKeys.front().display.size == 7 * previous.mvScaleFactors[previous.mvKeys.front().octave]);
    REQUIRE(previous.mvKeys.front().display.angle == -1);
    const ORB_SLAM3::Frame copied(previous);
    REQUIRE(copied.mvKeys.front().display.size == previous.mvKeys.front().display.size);
    REQUIRE(copied.mvKeys.front().pt_undistorted == previous.mvKeys.front().pt_undistorted);
    REQUIRE(copied.mvKeysRight.size() == previous.mvKeysRight.size());
    REQUIRE_FALSE(previous.mvKeys.front().descriptor.empty());
    previous.get_optical_flow_pyramid(image);
    previous.flow_candidates = 200;
    const auto       next_id = ORB_SLAM3::Frame::nNextId;
    ORB_SLAM3::Frame current(previous, 1.05);
    REQUIRE(current.mnId == next_id);
    REQUIRE(ORB_SLAM3::Frame::nNextId == next_id + 1);
    REQUIRE(current.flow_only);
    REQUIRE(current.N == 0);
    REQUIRE(current.mvKeys.empty());
    REQUIRE(current.mvKeysRight.empty());
    REQUIRE(current.optical_flow_pyramid.empty());
    REQUIRE(current.flow_candidates == 0);
    REQUIRE(current.mpPrevFrame == &previous);
    REQUIRE_FALSE(current.HasPose());
    REQUIRE_FALSE(current.imuIsPreintegrated());

    ORB_SLAM3::MapPoint point;
    cv::KeyPoint        tracked(73.25f, 81.75f, 31.0f);
    const cv::KeyPoint  native_tracked(tracked.pt, 7.f, -1.f);
    ORB_SLAM3::keypoint tracked_feature(tracked, native_tracked);
    tracked_feature.pt_undistorted   = cv::Point2f(-10, -20);
    tracked_feature.descriptor       = cv::Mat::ones(1, 32, CV_8U);
    tracked_feature.place_descriptor = cv::Mat::ones(1, 32, CV_8U);
    current.set_tracked_features({tracked_feature}, {&point});
    current.flow_candidates = 100;
    current.flow_tracks     = 1;
    REQUIRE(current.N == 1);
    REQUIRE(current.mvKeys.front().pt == tracked.pt);
    REQUIRE(current.mvKeys.front().display.pt == tracked.pt);
    REQUIRE(current.mvKeys.front().display.size == 7.f);
    REQUIRE(current.mvKeys.front().display.angle == -1.f);
    REQUIRE(current.mvKeys.front().size == 31.f);
    REQUIRE(current.mvKeys.front().descriptor.empty());
    REQUIRE(current.mvKeys.front().place_descriptor.empty());
    REQUIRE(current.mvKeysRight.empty());
    std::vector<cv::Point2f> expected_undistorted;
    cv::undistortPoints(std::vector<cv::Point2f>{tracked.pt}, expected_undistorted, calibration, distortion, cv::noArray(), calibration);
    REQUIRE(cv::norm(current.mvKeys.front().pt_undistorted - expected_undistorted.front()) < 0.001);
    REQUIRE(current.mvpMapPoints.front() == &point);
    REQUIRE(current.mvuRight.front() == -1);
    REQUIRE(current.mvDepth.front() == -1);
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
    ORB_SLAM3::Tracking::extract_features(current, left_extractor, image, cameras, stereo ? image : cv::Mat(), stereo ? &right_extractor : nullptr);
    REQUIRE_FALSE(current.flow_only);
    REQUIRE(current.flow_candidates == 100);
    REQUIRE(current.flow_tracks == 1);
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
    auto current_descriptors = descriptor_matrix(current.mvKeys);
    if (current.Nleft >= 0)
    {
        current_descriptors.push_back(descriptor_matrix(current.mvKeysRight));
    }
    auto previous_descriptors = descriptor_matrix(previous.mvKeys);
    if (previous.Nleft >= 0)
    {
        previous_descriptors.push_back(descriptor_matrix(previous.mvKeysRight));
    }
    REQUIRE(current_descriptors.rows == current.N);
    REQUIRE(cv::norm(current_descriptors, previous_descriptors, cv::NORM_INF) == 0);
    REQUIRE(current.mvpMapPoints.size() == static_cast<std::size_t>(current.N));
    REQUIRE(std::all_of(current.mvpMapPoints.begin(), current.mvpMapPoints.end(), [](const auto *entry) { return entry == nullptr; }));
    REQUIRE(current.mvKeys.size() == previous.mvKeys.size());
    REQUIRE(current.mvKeysRight.size() == previous.mvKeysRight.size());
    if (camera_kind == 2)
    {
        REQUIRE(std::all_of(current.mvKeys.begin(), current.mvKeys.end(), [](const auto &key) { return key.pt_undistorted == key.pt; }));
        REQUIRE(std::all_of(current.mvKeysRight.begin(), current.mvKeysRight.end(), [](const auto &key) { return key.pt_undistorted == key.pt; }));
    }
    REQUIRE(current.mvKeys.front().display.size == previous.mvKeys.front().display.size);
    REQUIRE(current.mvKeys.front().display.angle == previous.mvKeys.front().display.angle);
    REQUIRE(current.mvLeftToRightMatch == previous.mvLeftToRightMatch);
    REQUIRE(current.mvDepth == previous.mvDepth);
    REQUIRE(current.mBowVec.empty());
    REQUIRE(current.mFeatVec.empty());
    const auto first_grid = current.GetFeaturesInArea(128, 128, 256);
    ORB_SLAM3::Tracking::extract_features(current, left_extractor, image, cameras, stereo ? image : cv::Mat(), stereo ? &right_extractor : nullptr);
    REQUIRE(current.GetFeaturesInArea(128, 128, 256) == first_grid);

    const cv::Mat blank = cv::Mat::zeros(image.size(), image.type());
    ORB_SLAM3::Tracking::extract_features(current, left_extractor, blank, cameras, stereo ? blank : cv::Mat(), stereo ? &right_extractor : nullptr);
    REQUIRE(current.N == 0);
    REQUIRE(current.mvKeys.empty());
    REQUIRE(current.mvKeysRight.empty());
    REQUIRE(current.mvpMapPoints.empty());
    REQUIRE(current.mvDepth.empty());
    REQUIRE(current.GetFeaturesInArea(128, 128, 256).empty());
    if (camera_kind == 2)
    {
        REQUIRE(current.Nleft == 0);
        REQUIRE(current.Nright == 0);
        current.set_tracked_features({tracked}, {&point});
        REQUIRE(current.Nleft == 1);
        REQUIRE(current.Nright == 0);
        REQUIRE(current.mvKeys.front().pt_undistorted == tracked.pt);
        REQUIRE(current.mvLeftToRightMatch == std::vector<int>{-1});
    }
}

TEST_CASE("Stereo refresh restores dense observations and measured depth after flow")
{
    const bool fisheye = GENERATE(false, true);
    struct restore_threads
    {
        int count{cv::getNumThreads()};
        ~restore_threads() { cv::setNumThreads(count); }
    } restore;
    cv::setNumThreads(1);
    cv::Mat left(256, 256, CV_8UC1);
    cv::RNG random(27);
    random.fill(left, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(left, left, cv::Size(3, 3), 0);
    cv::Mat       right;
    const cv::Mat shift = cv::Mat(cv::Matx23d(1, 0, -4, 0, 1, 0));
    cv::warpAffine(left, right, shift, left.size());
    ORB_SLAM3::keypoint_extractor left_extractor(400, 1.2f, 4, 20, 7);
    ORB_SLAM3::keypoint_extractor right_extractor(400, 1.2f, 4, 20, 7);
    ORB_SLAM3::camera_vector      cameras;
    for (int camera_index = 0; camera_index < 2; ++camera_index)
    {
        if (fisheye)
        {
            auto camera           = std::make_unique<ORB_SLAM3::KannalaBrandt8>(std::vector<float>{200, 200, 128, 128, 0, 0, 0, 0});
            camera->mvLappingArea = {0, 256};
            cameras.push_back(std::move(camera));
        }
        else
        {
            cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{200, 200, 128, 128}));
        }
    }
    auto    calibration                     = cameras[0]->toK();
    cv::Mat distortion                      = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::Frame::mbInitialComputations = true;
    const Sophus::SE3f relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0, 0));
    ORB_SLAM3::Frame   previous(1.0, nullptr, calibration, distortion, 20, 40, nullptr, {}, relative_pose);
    ORB_SLAM3::Tracking::extract_features(previous, left_extractor, left, cameras, right, &right_extractor);
    previous.mpLastKeyFrame = nullptr;
    previous.mnDataset      = 0;
    ORB_SLAM3::Frame current(previous, 1.05);
    REQUIRE(current.flow_only);
    REQUIRE(current.mvKeysRight.empty());
    cv::setNumThreads(4);
    ORB_SLAM3::Tracking::extract_features(current, left_extractor, left, cameras, right, &right_extractor);
    REQUIRE_FALSE(current.flow_only);
    REQUIRE(current.mvKeys.size() == previous.mvKeys.size());
    REQUIRE(current.mvKeysRight.size() == previous.mvKeysRight.size());
    REQUIRE(current.mvKeys.size() > 300);
    REQUIRE(current.mvKeysRight.size() > 300);
    REQUIRE(std::count_if(current.mvDepth.begin(), current.mvDepth.end(), [](float depth) { return depth > 0; }) >= 50);
    REQUIRE(current.mvDepth == previous.mvDepth);
    REQUIRE(current.mvLeftToRightMatch == previous.mvLeftToRightMatch);
    REQUIRE(current.mvRightToLeftMatch == previous.mvRightToLeftMatch);
    if (fisheye)
    {
        const auto valid       = std::find_if(current.mvDepth.begin(), current.mvDepth.end(), [](float depth) { return depth > 0; });
        const auto left_index  = static_cast<size_t>(valid - current.mvDepth.begin());
        const auto right_index = current.mvLeftToRightMatch[left_index];
        current.mvKeys.push_back(current.mvKeys[left_index]);
        ++current.Nleft;
        ++current.N;
        current.ComputeStereoFishEyeMatches(cameras);
        REQUIRE(current.mvLeftToRightMatch.back() == right_index);
        REQUIRE(current.mvRightToLeftMatch[right_index] == current.Nleft - 1);
    }
}

TEST_CASE("RGB-D frames undistort tracked features using the primary camera")
{
    cv::Mat image(256, 256, CV_8UC1);
    cv::RNG random(17);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 4, 20, 7);
    cv::Mat                       calibration = cv::Mat(cv::Matx33f(200, 0, 128, 0, 200, 128, 0, 0, 1));
    cv::Mat                       distortion  = cv::Mat(cv::Vec4f(-0.1f, 0, 0, 0));
    const cv::Mat                 depth(image.size(), CV_32F, cv::Scalar(2));
    ORB_SLAM3::Frame::mbInitialComputations = true;
    ORB_SLAM3::Frame         frame(1.0, nullptr, calibration, distortion, 20, 40);
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{200, 200, 128, 128}));
    ORB_SLAM3::Tracking::extract_features(frame, extractor, image, cameras);
    frame.ComputeStereoFromRGBD(depth);
    REQUIRE(frame.N > 0);
    REQUIRE(frame.Nleft == -1);
    REQUIRE(std::all_of(frame.mvDepth.begin(), frame.mvDepth.end(), [](float value) { return value == 2.0f; }));
    REQUIRE(std::all_of(frame.mvuRight.begin(), frame.mvuRight.end(), [](float value) { return value >= 0; }));

    ORB_SLAM3::MapPoint point;
    const cv::KeyPoint  tracked(73.25f, 81.75f, 31.0f);
    frame.set_tracked_features({tracked}, {&point});
    std::vector<cv::Point2f> expected;
    cv::undistortPoints(std::vector<cv::Point2f>{tracked.pt}, expected, calibration, distortion, cv::noArray(), calibration);
    REQUIRE(cv::norm(frame.mvKeys.front().pt_undistorted - expected.front()) < 0.001);
    REQUIRE(frame.Nleft == -1);
}

TEST_CASE("Flow quality requires geometric inliers and spatial coverage")
{
    ORB_SLAM3::Frame frame;
    frame.mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::MapPoint              point;
    std::vector<ORB_SLAM3::keypoint> keys;
    for (int i = 0; i < 100; ++i)
    {
        keys.emplace_back(static_cast<float>(16 + (i % 4) * 32), static_cast<float>(16 + ((i / 4) % 4) * 32), 31.0f);
    }
    frame.set_tracked_features(keys, std::vector<ORB_SLAM3::MapPoint *>(100, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    REQUIRE(frame.flow_inliers == 100);
    REQUIRE(frame.flow_occupied_cells == 16);
    REQUIRE(frame.reliable_flow_inliers({128, 128}, 100));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}, 101));
    std::fill(frame.mvbOutlier.begin(), frame.mvbOutlier.begin() + 40, true);
    REQUIRE(frame.reliable_flow_inliers({128, 128})); // exactly 60%
    frame.mvbOutlier[40] = true;
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<ORB_SLAM3::keypoint>(100, ORB_SLAM3::keypoint(16, 16, 31)), std::vector<ORB_SLAM3::MapPoint *>(100, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<ORB_SLAM3::keypoint>(keys.begin(), keys.begin() + 49), std::vector<ORB_SLAM3::MapPoint *>(49, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<ORB_SLAM3::keypoint>(keys.begin(), keys.begin() + 50), std::vector<ORB_SLAM3::MapPoint *>(50, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    std::vector<ORB_SLAM3::keypoint> six_cells;
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