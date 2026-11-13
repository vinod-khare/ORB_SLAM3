#include "frame.h"
#include "kannala_brandt8.h"
#include "keypoint_extractor.h"
#include "map_point.h"
#include "optimizer.h"
#include "pinhole.h"
#include "tracking.h"

#include <algorithm>
#include <limits>
#include <memory>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEST_CASE("Default and copied frames own a valid IMU integration lock")
{
    ORB_SLAM3::frame frame;
    REQUIRE_FALSE(frame.imu_is_preintegrated());

    frame.set_integrated();
    REQUIRE(frame.imu_is_preintegrated());

    frame.calibration_matrix = cv::Mat::eye(3, 3, CV_32F);
    const ORB_SLAM3::frame copied(frame);
    REQUIRE(copied.imu_is_preintegrated());
}

TEST_CASE("Frame construction initializes state without extraction")
{
    const cv::Mat    calibration = cv::Mat(cv::Matx33f(200, 0, 128, 0, 200, 128, 0, 0, 1));
    const cv::Mat    distortion  = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::frame previous(1.0, nullptr, calibration, distortion, 20, 40);
    previous.set_velocity(Eigen::Vector3f(1, 2, 3));
    const Sophus::SE3f relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0, 0));
    const auto         next_id = ORB_SLAM3::frame::next_id;
    ORB_SLAM3::frame   current(1.05, nullptr, calibration, distortion, 20, 40, &previous, {}, relative_pose);
    REQUIRE(current.id == next_id);
    REQUIRE(ORB_SLAM3::frame::next_id == next_id + 1);
    REQUIRE(current.timestamp == 1.05);
    REQUIRE(current.feature_count == 0);
    REQUIRE(current.keypoints.empty());
    REQUIRE(current.right_keypoints.empty());
    REQUIRE(current.scale_factors.empty());
    REQUIRE(current.optical_flow_pyramid.empty());
    REQUIRE(current.previous_frame == &previous);
    REQUIRE(current.has_velocity());
    REQUIRE(current.get_velocity().isApprox(previous.get_velocity()));
    REQUIRE(current.get_relative_pose_tlr().matrix().isApprox(relative_pose.matrix()));
    REQUIRE_FALSE(current.has_pose());
    REQUIRE_FALSE(current.imu_is_preintegrated());
    REQUIRE(cv::norm(current.calibration_matrix, calibration, cv::NORM_INF) == 0);
    REQUIRE(current.calibration_matrix.data != calibration.data);
}

TEST_CASE("Frame feature data survives extractor reuse and destruction")
{
    const auto descriptor_type = GENERATE(ORB_SLAM3::keypoint_descriptor_type::orb, ORB_SLAM3::keypoint_descriptor_type::brisk, ORB_SLAM3::keypoint_descriptor_type::sift);
    cv::Mat    image(256, 256, CV_8UC1);
    cv::RNG    random(17);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 200, 128, 128}));
    const cv::Mat    distortion = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::frame frame(1.0, nullptr, cameras[0]->to_k(), distortion, 20, 40);
    ORB_SLAM3::frame::initial_computations = true;
    cv::Mat descriptors;
    {
        ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 4, 20, 7, ORB_SLAM3::keypoint_detector_type::fast, descriptor_type);
        ORB_SLAM3::tracking::extract_features(frame, extractor, image, cameras);
        REQUIRE(frame.feature_count > 0);
        REQUIRE(frame.scale_levels == extractor.get_levels());
        REQUIRE(frame.scale_factor == extractor.get_scale_factor());
        REQUIRE(frame.scale_factors == extractor.get_scale_factors());
        REQUIRE(frame.inverse_scale_factors == extractor.get_inverse_scale_factors());
        REQUIRE(frame.level_sigma_squared == extractor.get_scale_sigma_squares());
        REQUIRE(frame.inverse_level_sigma_squared == extractor.get_inverse_scale_sigma_squares());
        descriptors = descriptor_matrix(frame.keypoints).clone();
        ORB_SLAM3::frame other(2.0, nullptr, cameras[0]->to_k(), distortion, 20, 40);
        ORB_SLAM3::tracking::extract_features(other, extractor, cv::Mat::zeros(image.size(), image.type()), cameras);
        REQUIRE(other.feature_count == 0);
    }
    REQUIRE(cv::norm(descriptor_matrix(frame.keypoints), descriptors, cv::NORM_INF) == 0);
    const ORB_SLAM3::frame copied(frame);
    REQUIRE(copied.feature_count == frame.feature_count);
    REQUIRE(cv::norm(descriptor_matrix(copied.keypoints), descriptors, cv::NORM_INF) == 0);
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

    ORB_SLAM3::frame previous_frame;
    ORB_SLAM3::frame current_frame;
    const auto      &previous_pyramid = previous_frame.get_optical_flow_pyramid(previous_image);
    const auto      &current_pyramid  = current_frame.get_optical_flow_pyramid(current_image);

    REQUIRE(previous_pyramid.size() == 6);
    REQUIRE(previous_frame.get_optical_flow_pyramid(previous_image).front().data == previous_pyramid.front().data);
    REQUIRE(previous_pyramid.front().data != previous_image.data);
    REQUIRE(cv::norm(previous_pyramid.front(), previous_image, cv::NORM_INF) == 0);
    const auto &right_pyramid = previous_frame.get_optical_flow_pyramid(current_image, true);
    REQUIRE(cv::norm(right_pyramid.front(), current_image, cv::NORM_INF) == 0);
    REQUIRE(right_pyramid.front().data != previous_pyramid.front().data);
    previous_frame.calibration_matrix = cv::Mat::eye(3, 3, CV_32F);
    const ORB_SLAM3::frame copied(previous_frame);
    REQUIRE(copied.right_optical_flow_pyramid.front().data == right_pyramid.front().data);

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
            auto camera          = std::make_unique<ORB_SLAM3::kannala_brandt8>(std::vector<float>{200, 200, 128, 128, 0, 0, 0, 0});
            camera->lapping_area = {0, 256};
            cameras.push_back(std::move(camera));
        }
        else
        {
            cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 200, 128, 128}));
        }
    }

    auto    calibration = cameras[0]->to_k();
    cv::Mat distortion  = cv::Mat::zeros(4, 1, CV_32F);

    if (camera_kind == 3)
    {
        distortion.at<float>(0) = -0.1f;
    }
    Sophus::SE3f relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0, 0));
    ORB_SLAM3::frame::initial_computations = true;
    ORB_SLAM3::frame previous(1.0, nullptr, calibration, distortion, 20, 40, nullptr, {}, relative_pose);
    ORB_SLAM3::tracking::extract_features(previous, left_extractor, image, cameras, stereo ? image : cv::Mat(), stereo ? &right_extractor : nullptr);
    previous.last_key_frame = nullptr;
    previous.dataset        = 0;
    REQUIRE(previous.feature_count > 0);
    REQUIRE(previous.keypoints.front().size == 7 * previous.scale_factors[previous.keypoints.front().octave]);
    REQUIRE(previous.keypoints.front().angle == -1);
    const ORB_SLAM3::frame copied(previous);
    REQUIRE(copied.keypoints.front().size == previous.keypoints.front().size);
    REQUIRE(copied.keypoints.front().tracking_angle == previous.keypoints.front().tracking_angle);
    REQUIRE(copied.keypoints.front().pt_undistorted == previous.keypoints.front().pt_undistorted);
    REQUIRE(copied.right_keypoints.size() == previous.right_keypoints.size());
    REQUIRE_FALSE(previous.keypoints.front().descriptor.empty());
    previous.get_optical_flow_pyramid(image);
    previous.get_optical_flow_pyramid(image, true);
    previous.flow_candidates = 200;
    const auto       next_id = ORB_SLAM3::frame::next_id;
    ORB_SLAM3::frame current(previous, 1.05);
    REQUIRE(current.id == next_id);
    REQUIRE(ORB_SLAM3::frame::next_id == next_id + 1);
    REQUIRE(current.flow_only);
    REQUIRE(current.feature_count == 0);
    REQUIRE(current.keypoints.empty());
    REQUIRE(current.right_keypoints.empty());
    REQUIRE(current.optical_flow_pyramid.empty());
    REQUIRE(current.right_optical_flow_pyramid.empty());
    REQUIRE(current.flow_candidates == 0);
    REQUIRE(current.previous_frame == &previous);
    REQUIRE_FALSE(current.has_pose());
    REQUIRE_FALSE(current.imu_is_preintegrated());

    ORB_SLAM3::map_point point;
    cv::KeyPoint         tracked(73.25f, 81.75f, 31.0f);
    const cv::KeyPoint   native_tracked(tracked.pt, 7.f, -1.f);
    ORB_SLAM3::keypoint  tracked_feature(native_tracked, 42.f);
    tracked_feature.pt_undistorted = cv::Point2f(-10, -20);
    tracked_feature.descriptor     = cv::Mat::ones(1, 32, CV_8U);
    current.set_tracked_features({tracked_feature}, {&point});
    current.flow_candidates = 100;
    current.flow_tracks     = 1;
    REQUIRE(current.feature_count == 1);
    REQUIRE(current.keypoints.front().pt == tracked.pt);
    REQUIRE(current.keypoints.front().size == 7.f);
    REQUIRE(current.keypoints.front().angle == -1.f);
    REQUIRE(current.keypoints.front().tracking_angle == 42.f);
    REQUIRE(current.keypoints.front().descriptor.empty());
    REQUIRE(current.right_keypoints.empty());
    std::vector<cv::Point2f> expected_undistorted;
    cv::undistortPoints(std::vector<cv::Point2f>{tracked.pt}, expected_undistorted, calibration, distortion, cv::noArray(), calibration);
    REQUIRE(cv::norm(current.keypoints.front().pt_undistorted - expected_undistorted.front()) < 0.001);
    REQUIRE(current.map_points.front() == &point);
    REQUIRE(current.right_coordinates.front() == -1);
    REQUIRE(current.depths.front() == -1);
    REQUIRE(current.is_fisheye_stereo == (camera_kind == 2));
    REQUIRE(current.right_keypoints.empty());
    if (camera_kind == 2)
    {
        REQUIRE(current.right_keypoints.empty());
        REQUIRE(current.left_to_right_matches == std::vector<int>{-1});
    }

    const auto pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f(1, 2, 3));
    current.set_pose(pose);
    current.set_velocity(Eigen::Vector3f(4, 5, 6));
    ORB_SLAM3::imu::preintegrated frame_integration;
    ORB_SLAM3::imu::preintegrated keyframe_integration;
    current.frame_imu_preintegrated = &frame_integration;
    current.imu_preintegrated       = &keyframe_integration;
    current.set_integrated();
    current.get_optical_flow_pyramid(image);
    const auto *pyramid_data = current.optical_flow_pyramid.front().data;
    ORB_SLAM3::tracking::extract_features(current, left_extractor, image, cameras, stereo ? image : cv::Mat(), stereo ? &right_extractor : nullptr);
    REQUIRE_FALSE(current.flow_only);
    REQUIRE(current.flow_candidates == 100);
    REQUIRE(current.flow_tracks == 1);
    REQUIRE(current.id == next_id);
    REQUIRE(ORB_SLAM3::frame::next_id == next_id + 1);
    REQUIRE(current.timestamp == 1.05);
    REQUIRE(current.previous_frame == &previous);
    REQUIRE(current.imu_is_preintegrated());
    REQUIRE(current.frame_imu_preintegrated == &frame_integration);
    REQUIRE(current.imu_preintegrated == &keyframe_integration);
    REQUIRE(current.get_pose().matrix().isApprox(pose.matrix()));
    REQUIRE(current.get_velocity().isApprox(Eigen::Vector3f(4, 5, 6)));
    REQUIRE(current.optical_flow_pyramid.front().data == pyramid_data);
    REQUIRE(current.feature_count == previous.feature_count);
    auto current_descriptors = descriptor_matrix(current.keypoints);
    if (current.is_fisheye_stereo)
    {
        current_descriptors.push_back(descriptor_matrix(current.right_keypoints));
    }
    auto previous_descriptors = descriptor_matrix(previous.keypoints);
    if (previous.is_fisheye_stereo)
    {
        previous_descriptors.push_back(descriptor_matrix(previous.right_keypoints));
    }
    REQUIRE(current_descriptors.rows == current.feature_count);
    REQUIRE(cv::norm(current_descriptors, previous_descriptors, cv::NORM_INF) == 0);
    REQUIRE(current.map_points.size() == static_cast<std::size_t>(current.feature_count));
    REQUIRE(std::all_of(current.map_points.begin(), current.map_points.end(), [](const auto *entry) { return entry == nullptr; }));
    REQUIRE(current.keypoints.size() == previous.keypoints.size());
    REQUIRE(current.right_keypoints.size() == previous.right_keypoints.size());
    if (camera_kind == 2)
    {
        REQUIRE(std::all_of(current.keypoints.begin(), current.keypoints.end(), [](const auto &key) { return key.pt_undistorted == key.pt; }));
        REQUIRE(std::all_of(current.right_keypoints.begin(), current.right_keypoints.end(), [](const auto &key) { return key.pt_undistorted == key.pt; }));
    }
    REQUIRE(current.keypoints.front().size == previous.keypoints.front().size);
    REQUIRE(current.keypoints.front().angle == previous.keypoints.front().angle);
    REQUIRE(current.keypoints.front().tracking_angle == previous.keypoints.front().tracking_angle);
    REQUIRE(current.left_to_right_matches == previous.left_to_right_matches);
    REQUIRE(current.depths == previous.depths);
    REQUIRE(current.bow_vector.empty());
    REQUIRE(current.feature_vector.empty());
    const auto first_grid = current.get_features_in_area(128, 128, 256);
    ORB_SLAM3::tracking::extract_features(current, left_extractor, image, cameras, stereo ? image : cv::Mat(), stereo ? &right_extractor : nullptr);
    REQUIRE(current.get_features_in_area(128, 128, 256) == first_grid);

    const cv::Mat blank = cv::Mat::zeros(image.size(), image.type());
    ORB_SLAM3::tracking::extract_features(current, left_extractor, blank, cameras, stereo ? blank : cv::Mat(), stereo ? &right_extractor : nullptr);
    REQUIRE(current.feature_count == 0);
    REQUIRE(current.keypoints.empty());
    REQUIRE(current.right_keypoints.empty());
    REQUIRE(current.map_points.empty());
    REQUIRE(current.depths.empty());
    REQUIRE(current.get_features_in_area(128, 128, 256).empty());
    if (camera_kind == 2)
    {
        REQUIRE(current.is_fisheye_stereo);
        REQUIRE(current.keypoints.empty());
        REQUIRE(current.right_keypoints.empty());
        current.set_tracked_features({tracked}, {&point});
        REQUIRE(current.is_fisheye_stereo);
        REQUIRE(current.keypoints.size() == 1);
        REQUIRE(current.right_keypoints.empty());
        REQUIRE(current.keypoints.front().pt_undistorted == tracked.pt);
        REQUIRE(current.left_to_right_matches == std::vector<int>{-1});
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
            auto camera          = std::make_unique<ORB_SLAM3::kannala_brandt8>(std::vector<float>{200, 200, 128, 128, 0, 0, 0, 0});
            camera->lapping_area = {0, 256};
            cameras.push_back(std::move(camera));
        }
        else
        {
            cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 200, 128, 128}));
        }
    }
    auto    calibration                    = cameras[0]->to_k();
    cv::Mat distortion                     = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::frame::initial_computations = true;
    const Sophus::SE3f relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0, 0));
    ORB_SLAM3::frame   previous(1.0, nullptr, calibration, distortion, 20, 40, nullptr, {}, relative_pose);
    ORB_SLAM3::tracking::extract_features(previous, left_extractor, left, cameras, right, &right_extractor);
    previous.last_key_frame = nullptr;
    previous.dataset        = 0;
    ORB_SLAM3::frame current(previous, 1.05);
    REQUIRE(current.flow_only);
    REQUIRE(current.right_keypoints.empty());
    cv::setNumThreads(4);
    ORB_SLAM3::tracking::extract_features(current, left_extractor, left, cameras, right, &right_extractor);
    REQUIRE_FALSE(current.flow_only);
    REQUIRE(current.keypoints.size() == previous.keypoints.size());
    REQUIRE(current.right_keypoints.size() == previous.right_keypoints.size());
    REQUIRE(current.keypoints.size() > 300);
    REQUIRE(current.right_keypoints.size() > 300);
    REQUIRE(std::count_if(current.depths.begin(), current.depths.end(), [](float depth) { return depth > 0; }) >= 50);
    REQUIRE(current.depths == previous.depths);
    REQUIRE(current.left_to_right_matches == previous.left_to_right_matches);
    REQUIRE(current.right_to_left_matches == previous.right_to_left_matches);
    if (fisheye)
    {
        const auto valid       = std::find_if(current.depths.begin(), current.depths.end(), [](float depth) { return depth > 0; });
        const auto left_index  = static_cast<size_t>(valid - current.depths.begin());
        const auto right_index = current.left_to_right_matches[left_index];
        current.keypoints.push_back(current.keypoints[left_index]);
        ++current.feature_count;
        current.compute_stereo_fisheye_matches(cameras);
        REQUIRE(current.left_to_right_matches.back() == right_index);
        REQUIRE(current.right_to_left_matches[right_index] == static_cast<int>(current.keypoints.size()) - 1);
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
    ORB_SLAM3::frame::initial_computations = true;
    ORB_SLAM3::frame         frame(1.0, nullptr, calibration, distortion, 20, 40);
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 200, 128, 128}));
    ORB_SLAM3::tracking::extract_features(frame, extractor, image, cameras);
    frame.compute_stereo_from_rgbd(depth);
    REQUIRE(frame.feature_count > 0);
    REQUIRE_FALSE(frame.is_fisheye_stereo);
    REQUIRE(std::all_of(frame.depths.begin(), frame.depths.end(), [](float value) { return value == 2.0f; }));
    REQUIRE(std::all_of(frame.right_coordinates.begin(), frame.right_coordinates.end(), [](float value) { return value >= 0; }));

    ORB_SLAM3::map_point point;
    const cv::KeyPoint   tracked(73.25f, 81.75f, 31.0f);
    frame.set_tracked_features({tracked}, {&point});
    std::vector<cv::Point2f> expected;
    cv::undistortPoints(std::vector<cv::Point2f>{tracked.pt}, expected, calibration, distortion, cv::noArray(), calibration);
    REQUIRE(cv::norm(frame.keypoints.front().pt_undistorted - expected.front()) < 0.001);
    REQUIRE_FALSE(frame.is_fisheye_stereo);
}

TEST_CASE("Flow quality requires geometric inliers and spatial coverage")
{
    ORB_SLAM3::frame frame;
    frame.distortion_coefficients = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::map_point             point;
    std::vector<ORB_SLAM3::keypoint> keys;
    for (int i = 0; i < 100; ++i)
    {
        keys.emplace_back(static_cast<float>(16 + (i % 4) * 32), static_cast<float>(16 + ((i / 4) % 4) * 32), 31.0f);
    }
    frame.set_tracked_features(keys, std::vector<ORB_SLAM3::map_point *>(100, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    REQUIRE(frame.flow_inliers == 100);
    REQUIRE(frame.flow_occupied_cells == 16);
    REQUIRE(frame.reliable_flow_inliers({128, 128}, 100));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}, 101));
    std::fill(frame.outliers.begin(), frame.outliers.begin() + 40, true);
    REQUIRE(frame.reliable_flow_inliers({128, 128})); // exactly 60%
    frame.outliers[40] = true;
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<ORB_SLAM3::keypoint>(100, ORB_SLAM3::keypoint(16, 16, 31)), std::vector<ORB_SLAM3::map_point *>(100, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<ORB_SLAM3::keypoint>(keys.begin(), keys.begin() + 49), std::vector<ORB_SLAM3::map_point *>(49, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(std::vector<ORB_SLAM3::keypoint>(keys.begin(), keys.begin() + 50), std::vector<ORB_SLAM3::map_point *>(50, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    std::vector<ORB_SLAM3::keypoint> six_cells;
    for (int i = 0; i < 60; ++i)
    {
        six_cells.emplace_back(static_cast<float>(16 + (i % 3) * 32), static_cast<float>(16 + ((i / 3) % 2) * 32), 31.0f);
    }
    frame.set_tracked_features(six_cells, std::vector<ORB_SLAM3::map_point *>(60, &point));
    REQUIRE(frame.reliable_flow_inliers({128, 128}));
    for (auto &key : six_cells)
    {
        if (key.pt == cv::Point2f(80, 48))
        {
            key.pt = cv::Point2f(16, 16);
        }
    }
    frame.set_tracked_features(six_cells, std::vector<ORB_SLAM3::map_point *>(60, &point));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    frame.set_tracked_features(keys, std::vector<ORB_SLAM3::map_point *>(100, nullptr));
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    for (auto &key : keys)
    {
        key.pt.x = std::numeric_limits<float>::quiet_NaN();
    }
    frame.keypoints = keys;
    frame.map_points.assign(100, &point);
    REQUIRE_FALSE(frame.reliable_flow_inliers({128, 128}));
    REQUIRE_FALSE(frame.reliable_flow_inliers({0, 128}));
}

TEST_CASE("Fisheye flow preserves both camera observations for pose optimization")
{
    const bool               right_only = GENERATE(false, true);
    ORB_SLAM3::camera_vector cameras;
    for (int view = 0; view < 2; ++view)
    {
        auto camera          = std::make_unique<ORB_SLAM3::kannala_brandt8>(std::vector<float>{200, 200, 128, 128, 0, 0, 0, 0});
        camera->lapping_area = {0, 256};
        cameras.push_back(std::move(camera));
    }
    cv::Mat image(256, 256, CV_8UC1);
    cv::RNG random(27);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::keypoint_extractor left_extractor(400, 1.2f, 4, 20, 7);
    ORB_SLAM3::keypoint_extractor right_extractor(400, 1.2f, 4, 20, 7);
    const Sophus::SE3f            relative_pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.1f, 0.02f, 0));
    ORB_SLAM3::frame              frame(1.0, nullptr, cameras[0]->to_k(), cv::Mat::zeros(4, 1, CV_32F), 20, 40, nullptr, {}, relative_pose);
    ORB_SLAM3::frame::initial_computations = true;
    ORB_SLAM3::tracking::extract_features(frame, left_extractor, image, cameras, image, &right_extractor);

    std::vector<std::unique_ptr<ORB_SLAM3::map_point>> points;
    std::vector<ORB_SLAM3::keypoint>                   left_keys;
    std::vector<ORB_SLAM3::keypoint>                   right_keys;
    std::vector<ORB_SLAM3::map_point *>                left_points;
    std::vector<ORB_SLAM3::map_point *>                right_points;
    for (int index = 0; index < 100; ++index)
    {
        const Eigen::Vector3f position((index % 10 - 4.5f) * 0.3f, (index / 10 - 4.5f) * 0.3f, 3.0f + (index % 3) * 0.2f);
        auto                  point = std::make_unique<ORB_SLAM3::map_point>();
        point->set_world_position(position);
        const auto left  = cameras[0]->project(position);
        const auto right = cameras[1]->project(frame.get_relative_pose_trl() * position);
        if (!right_only)
        {
            left_keys.emplace_back(left.x(), left.y(), 7.f);
            left_points.push_back(point.get());
        }
        right_keys.emplace_back(right.x(), right.y(), 7.f);
        right_points.push_back(point.get());
        points.push_back(std::move(point));
    }
    frame.set_tracked_features(left_keys, left_points, right_keys, right_points);
    REQUIRE(frame.feature_count == (right_only ? 100 : 200));
    REQUIRE(frame.right_keypoints.size() == 100);
    REQUIRE(frame.map_points[frame.keypoints.size()] == right_points.front());
    REQUIRE(frame.outliers.size() == static_cast<std::size_t>(frame.feature_count));
    REQUIRE(frame.left_to_right_matches.size() == left_keys.size());
    REQUIRE(frame.right_to_left_matches == std::vector<int>(100, -1));
    REQUIRE(std::all_of(frame.depths.begin(), frame.depths.end(), [](float depth) { return depth == -1; }));
    REQUIRE(frame.get_features_in_area(128, 128, 128, -1, -1, true).size() == 100);
    REQUIRE(frame.reliable_flow_inliers({256, 256}, 100));
    REQUIRE(frame.flow_inliers == frame.feature_count);
    REQUIRE(frame.right_keypoints.front().descriptor.empty());
    REQUIRE(frame.right_keypoints.front().pt_undistorted == frame.right_keypoints.front().pt);

    frame.set_pose(Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.02f, -0.01f, 0.03f)));
    REQUIRE(ORB_SLAM3::optimizer::pose_optimization(&frame, cameras) == frame.feature_count);
    REQUIRE(frame.get_pose().translation().norm() < 0.001f);
    REQUIRE(frame.get_pose().so3().log().norm() < 0.001f);
    REQUIRE(frame.reliable_flow_inliers({256, 256}, 100));
    std::fill(frame.outliers.begin(), frame.outliers.end(), true);
    REQUIRE_FALSE(frame.reliable_flow_inliers({256, 256}));
}