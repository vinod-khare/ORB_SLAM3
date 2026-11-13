#include "descriptor_matcher.h"
#include "frame.h"
#include "g2o_types.h"
#include "kannala_brandt8.h"
#include "keypoint_extractor.h"
#include "map_point.h"
#include "pinhole.h"
#include "tracking.h"

#include <algorithm>
#include <memory>

#include <opencv2/xfeatures2d.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace
{

cv::Ptr<cv::Feature2D> reference_detector(ORB_SLAM3::keypoint_detector_type type)
{
    using detector = ORB_SLAM3::keypoint_detector_type;
    switch (type)
    {
    case detector::fast:
        return cv::FastFeatureDetector::create(20);
    case detector::agast:
        return cv::AgastFeatureDetector::create(20);
    case detector::gftt:
        return cv::GFTTDetector::create(0, 0.01, 1.0, 3, false);
    case detector::orb:
        return cv::ORB::create(400, 1.2f, 1, 19, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    case detector::brisk:
        return cv::BRISK::create(20, 0);
    case detector::sift:
        return cv::SIFT::create();
    case detector::akaze:
        return cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 1);
    case detector::star:
        return cv::xfeatures2d::StarDetector::create();
    case detector::harris_laplace:
        return cv::xfeatures2d::HarrisLaplaceFeatureDetector::create(1);
    case detector::msd:
        return cv::xfeatures2d::MSDDetector::create(3, 5, 5, 0, 250.0f, 4, 1.25f, 1);
    case detector::tbmr:
        return cv::xfeatures2d::TBMR::create(60, 0.01f, 1.25f, 1);
    case detector::surf:
        return cv::xfeatures2d::SURF::create(100, 1);
    }
    FAIL("Unknown detector type");
    return {};
}

std::pair<int, int> descriptor_layout(ORB_SLAM3::keypoint_descriptor_type type)
{
    using descriptor = ORB_SLAM3::keypoint_descriptor_type;
    switch (type)
    {
    case descriptor::orb:
    case descriptor::brief:
    case descriptor::latch:
    case descriptor::teblid:
    case descriptor::boostdesc:
        return {32, CV_8U};
    case descriptor::brisk:
    case descriptor::freak:
    case descriptor::beblid:
        return {64, CV_8U};
    case descriptor::sift:
        return {128, CV_32F};
    case descriptor::daisy:
        return {200, CV_32F};
    case descriptor::vgg:
        return {120, CV_32F};
    case descriptor::surf:
        return {64, CV_32F};
    }
    FAIL("Unknown descriptor type");
    return {};
}

} // namespace

TEST_CASE("Keypoints expose undistorted coordinates without changing image coordinates", "[undistorted_keypoints]")
{
    const cv::KeyPoint               tracked(73.25f, 81.75f, 31.f, 42.f, 0.75f, 2, 7);
    std::vector<ORB_SLAM3::keypoint> keys{
        ORB_SLAM3::keypoint{},
        ORB_SLAM3::keypoint(tracked.pt, tracked.size),
        ORB_SLAM3::keypoint(tracked.pt.x, tracked.pt.y, tracked.size),
        ORB_SLAM3::keypoint(tracked),
        ORB_SLAM3::keypoint(tracked, 15.f),
    };
    for (auto &key : keys)
    {
        REQUIRE(key.pt_undistorted == key.pt);
        const auto original  = key.pt;
        key.pt_undistorted  += cv::Point2f(2, 3);
        REQUIRE(key.pt == original);
    }
    const auto original    = ORB_SLAM3::to_cv_keypoints(keys);
    const auto undistorted = ORB_SLAM3::to_cv_keypoints(keys, true);
    REQUIRE(original.size() == keys.size());
    REQUIRE(undistorted.size() == keys.size());
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        REQUIRE(original[index].pt == keys[index].pt);
        REQUIRE(undistorted[index].pt == keys[index].pt_undistorted);
        REQUIRE(undistorted[index].size == keys[index].size);
        REQUIRE(undistorted[index].angle == keys[index].angle);
        REQUIRE(undistorted[index].response == keys[index].response);
        REQUIRE(undistorted[index].octave == keys[index].octave);
        REQUIRE(undistorted[index].class_id == keys[index].class_id);
    }
}

TEST_CASE("Frames and keyframes use embedded undistorted coordinates for geometry", "[undistorted_keypoints]")
{
    const auto               distortion_coefficient = GENERATE(0.f, -0.1f);
    const cv::Mat            calibration            = cv::Mat(cv::Matx33f(200, 0, 128, 0, 200, 128, 0, 0, 1));
    const cv::Mat            distortion             = cv::Mat(cv::Vec4f(distortion_coefficient, 0, 0, 0));
    const cv::Mat            image                  = cv::Mat::zeros(256, 256, CV_8U);
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 200, 128, 128}));
    cameras[0]->set_distortion_coefficients(distortion);
    cameras[0]->set_baseline(0.1f);
    ORB_SLAM3::frame    frame(0.0, nullptr, 40);
    ORB_SLAM3::keypoint key(73.25f, 81.75f, 31.f);
    key.pt_undistorted = cv::Point2f(-10, -20);
    key.descriptor     = cv::Mat::zeros(1, 32, CV_8U);
    frame.keypoints    = {key};
    frame.initialize_features(image, cameras);
    frame.set_pose(Sophus::SE3f{});
    std::vector<cv::Point2f> expected;
    cv::undistortPoints(std::vector<cv::Point2f>{key.pt}, expected, calibration, distortion, cv::noArray(), calibration);
    const auto &position = expected.front();
    REQUIRE(frame.keypoints.front().pt == key.pt);
    REQUIRE(cv::norm(frame.keypoints.front().pt_undistorted - position) < 0.001);
    REQUIRE(frame.get_features_in_area(position.x, position.y, 0.01f) == std::vector<std::size_t>{0});
    if (distortion_coefficient != 0)
    {
        REQUIRE(frame.get_features_in_area(key.pt.x, key.pt.y, 0.01f).empty());
    }

    cv::Mat depth                                                           = cv::Mat::zeros(image.size(), CV_32F);
    depth.at<float>(static_cast<int>(key.pt.y), static_cast<int>(key.pt.x)) = 2.f;
    frame.compute_stereo_from_rgbd(depth, *cameras[0]);
    REQUIRE(frame.depths.front() == 2.f);
    REQUIRE(frame.right_coordinates.front() == Catch::Approx(position.x - 10.f));
    Eigen::Vector3f point;
    REQUIRE(frame.unproject_stereo(0, point, *cameras[0]));
    REQUIRE(point.x() == Catch::Approx((position.x - 128.f) * 2.f / 200.f));
    REQUIRE(point.y() == Catch::Approx((position.y - 128.f) * 2.f / 200.f));
    REQUIRE(point.z() == 2.f);

    ORB_SLAM3::map         map;
    ORB_SLAM3::key_frame   keyframe(frame, &map, nullptr);
    const ORB_SLAM3::frame copied(frame);
    REQUIRE(keyframe.keypoints.front().pt == key.pt);
    REQUIRE(keyframe.keypoints.front().pt_undistorted == frame.keypoints.front().pt_undistorted);
    REQUIRE(copied.keypoints.front().pt_undistorted == frame.keypoints.front().pt_undistorted);
    REQUIRE(keyframe.get_features_in_area(position.x, position.y, 0.01f) == std::vector<std::size_t>{0});
    REQUIRE(keyframe.descriptor(0).data == frame.descriptor(0).data);

    ORB_SLAM3::descriptor_matcher matcher(0.9f, false);
    std::vector<cv::Point2f>      previous_matches{position};
    std::vector<int>              matches;
    REQUIRE(matcher.search_for_initialization(frame, frame, previous_matches, matches, 2) == 1);
    REQUIRE(matches == std::vector<int>{0});
    REQUIRE(previous_matches.front() == frame.keypoints.front().pt_undistorted);

    frame.set_tracked_features(*cameras[0], {}, {});
    REQUIRE(frame.keypoints.empty());
    REQUIRE(frame.get_features_in_area(position.x, position.y, 2).empty());
}

TEST_CASE("Initialization matching filters using tracking rather than native orientation", "[native_keypoints]")
{
    const cv::Mat            image = cv::Mat::zeros(256, 256, CV_8U);
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 200, 128, 128}));
    ORB_SLAM3::frame         first(0.0, nullptr, 40);
    ORB_SLAM3::frame         second(1.0, nullptr, 40);
    std::vector<cv::Point2f> positions;
    for (int index = 0; index < 12; ++index)
    {
        const cv::Point2f   position(30.f + 20.f * (index % 4), 30.f + 20.f * (index / 4));
        ORB_SLAM3::keypoint key(cv::KeyPoint(position, 7.f, -1.f), 0.f);
        key.descriptor = cv::Mat::zeros(1, 32, CV_8U);
        first.keypoints.push_back(key);
        second.keypoints.push_back(key);
        positions.push_back(position);
    }
    first.initialize_features(image, cameras);
    second.initialize_features(image, cameras);
    ORB_SLAM3::descriptor_matcher matcher(0.9f, true);
    std::vector<int>              matches;
    REQUIRE(matcher.search_for_initialization(first, second, positions, matches, 2) == 12);
    second.keypoints.back().tracking_angle = 180.f;
    REQUIRE(matcher.search_for_initialization(first, second, positions, matches, 2) == 11);
    REQUIRE(matches.back() == -1);
}

TEST_CASE("Frames use the camera supplied for each operation", "[camera_calibration]")
{
    ORB_SLAM3::camera_vector first_camera;
    ORB_SLAM3::camera_vector second_camera;
    first_camera.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{100, 100, 50, 50}));
    second_camera.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 200, 60, 60}));
    first_camera[0]->set_baseline(0.1f);
    second_camera[0]->set_baseline(0.2f);
    cv::Mat distortion = cv::Mat(cv::Vec4f(-0.2f, 0, 0, 0));
    second_camera[0]->set_distortion_coefficients(distortion);
    distortion.setTo(0);
    REQUIRE_THROWS(second_camera[0]->set_distortion_coefficients(cv::Mat::zeros(3, 1, CV_32F)));
    ORB_SLAM3::pinhole copied_camera(static_cast<ORB_SLAM3::pinhole *>(second_camera[0].get()));
    REQUIRE(copied_camera.baseline() == Catch::Approx(0.2f));
    REQUIRE(copied_camera.distortion_coefficients().at<float>(0) == Catch::Approx(-0.2f));
    REQUIRE(copied_camera.distortion_coefficients().data != second_camera[0]->distortion_coefficients().data);

    const cv::Mat    image = cv::Mat::zeros(120, 120, CV_8U);
    ORB_SLAM3::frame first(0, nullptr, 40);
    ORB_SLAM3::frame second(1, nullptr, 40);
    first.keypoints.emplace_back(30.f, 30.f, 7.f);
    second.keypoints.emplace_back(30.f, 30.f, 7.f);
    first.initialize_features(image, first_camera);
    second.initialize_features(image, second_camera);
    first.set_pose(Sophus::SE3f{});
    second.set_pose(Sophus::SE3f{});

    const cv::Mat depth(image.size(), CV_32F, cv::Scalar(2));
    first.compute_stereo_from_rgbd(depth, *first_camera[0]);
    second.compute_stereo_from_rgbd(depth, *second_camera[0]);
    REQUIRE(first.right_coordinates[0] == Catch::Approx(25));
    REQUIRE(second.right_coordinates[0] == Catch::Approx(second.keypoints[0].pt_undistorted.x - 20));
    REQUIRE(second.keypoints[0].pt_undistorted != second.keypoints[0].pt);
    REQUIRE(first.get_features_in_area(30, 30, 1) == std::vector<std::size_t>{0});

    Eigen::Vector3f point;
    REQUIRE(first.unproject_stereo(0, point, *first_camera[0]));
    REQUIRE(point.x() == Catch::Approx(-0.4f));
    REQUIRE(first.unproject_stereo(0, point, *second_camera[0]));
    REQUIRE(point.x() == Catch::Approx(-0.3f));

    ORB_SLAM3::map              map;
    const ORB_SLAM3::imu::calib imu_calibration(Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f(1, 2, 3)), 1, 1, 1, 1);
    ORB_SLAM3::key_frame        keyframe(first, &map, nullptr, imu_calibration);
    REQUIRE(keyframe.imu_calibration.camera_to_body.matrix().isApprox(imu_calibration.camera_to_body.matrix()));
}

TEST_CASE("Keyframe geometry uses explicitly supplied cameras", "[camera_calibration]")
{
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{100, 100, 50, 50}));
    cameras[0]->set_baseline(0.1f);
    ORB_SLAM3::frame frame(0, nullptr, 40);
    frame.keypoints.emplace_back(30.f, 40.f, 7.f);
    const cv::Mat image(160, 160, CV_8U, cv::Scalar(0));
    frame.initialize_features(image, cameras);
    frame.depths[0] = 2.f;
    const Sophus::SE3f pose(Sophus::SO3f::exp(Eigen::Vector3f(0.1f, -0.2f, 0.3f)), Eigen::Vector3f(1, 2, 3));
    frame.set_pose(pose);
    ORB_SLAM3::map       map;
    ORB_SLAM3::key_frame keyframe(frame, &map, nullptr);

    cameras.clear();
    cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{200, 250, 60, 70}));
    cameras[0]->set_baseline(0.2f);
    const int     coefficient_count = GENERATE(4, 5);
    const cv::Mat distortion        = cv::Mat(cv::Vec<float, 5>(-0.1f, 0.02f, 0.003f, -0.004f, 0.01f));
    cameras[0]->set_distortion_coefficients(distortion.rowRange(0, coefficient_count));

    Eigen::Vector3f unprojected;
    REQUIRE(keyframe.unproject_stereo(0, unprojected, *cameras[0]));
    REQUIRE(unprojected.isApprox(pose.inverse() * Eigen::Vector3f(-0.3f, -0.24f, 2.f)));

    const Eigen::Vector3f camera_point(0.2f, 0.3f, 2.f);
    ORB_SLAM3::map_point  point(pose.inverse() * camera_point, &keyframe, &map);
    cv::Point2f           projected;
    float                 u;
    float                 v;
    REQUIRE(keyframe.project_point_undistort(&point, projected, u, v, *cameras[0]));
    REQUIRE(u == Catch::Approx(80.f));
    REQUIRE(v == Catch::Approx(107.5f));
    REQUIRE(projected == cv::Point2f(u, v));

    const std::vector<cv::Point3f> object_points{cv::Point3f(0.2f, 0.3f, 2.f)};
    std::vector<cv::Point2f>       expected;
    cv::projectPoints(object_points, cv::Vec3f::all(0), cv::Vec3f::all(0), cameras[0]->to_k(), cameras[0]->distortion_coefficients(), expected);
    REQUIRE(keyframe.project_point_distort(&point, projected, u, v, *cameras[0]));
    REQUIRE(cv::norm(projected - expected.front()) < 0.001f);
    REQUIRE(projected == cv::Point2f(u, v));

    ORB_SLAM3::inverse_depth_point inverse_depth(0.5, 30, 40, *cameras[0]);
    const Eigen::Vector3d          reconstructed(
        (inverse_depth.u - inverse_depth.cx) / (inverse_depth.fx * inverse_depth.rho), (inverse_depth.v - inverse_depth.cy) / (inverse_depth.fy * inverse_depth.rho), 1 / inverse_depth.rho);
    REQUIRE(reconstructed.isApprox(Eigen::Vector3d(-0.3, -0.24, 2), 1e-6));
    REQUIRE(inverse_depth.bf == Catch::Approx(40.f));
    const ORB_SLAM3::imu_cam_pose optimizer_pose(&keyframe, cameras);
    REQUIRE(optimizer_pose.bf == Catch::Approx(40.f));

    point.set_world_position(pose.inverse() * Eigen::Vector3f(0, 0, -1));
    REQUIRE_FALSE(keyframe.project_point_undistort(&point, projected, u, v, *cameras[0]));
    REQUIRE_FALSE(keyframe.project_point_distort(&point, projected, u, v, *cameras[0]));
    point.set_world_position(pose.inverse() * Eigen::Vector3f(10, 0, 1));
    REQUIRE_FALSE(keyframe.project_point_undistort(&point, projected, u, v, *cameras[0]));
    REQUIRE_FALSE(keyframe.project_point_distort(&point, projected, u, v, *cameras[0]));

    ORB_SLAM3::pinhole alternate(std::vector<float>{100, 100, 50, 50});
    REQUIRE(keyframe.unproject_stereo(0, unprojected, alternate));
    REQUIRE(unprojected.isApprox(pose.inverse() * Eigen::Vector3f(-0.4f, -0.2f, 2.f)));
}

TEST_CASE("Keyframe stereo poses read camera extrinsics", "[camera_calibration]")
{
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::kannala_brandt8>(std::vector<float>{100, 100, 80, 80, 0, 0, 0, 0}));
    cameras.push_back(std::make_unique<ORB_SLAM3::kannala_brandt8>(std::vector<float>{110, 110, 80, 80, 0, 0, 0, 0}));
    ORB_SLAM3::frame frame(0, nullptr, 40);
    frame.keypoints.emplace_back(30.f, 40.f, 7.f);
    frame.right_keypoints.emplace_back(35.f, 45.f, 7.f);
    frame.keypoints[0].descriptor       = cv::Mat(1, 32, CV_8U, cv::Scalar(1));
    frame.right_keypoints[0].descriptor = cv::Mat(1, 32, CV_8U, cv::Scalar(2));
    frame.mono_left = frame.mono_right = 1;
    frame.scale_levels                 = 1;
    frame.scale_factors                = {1.f};
    const cv::Mat image(160, 160, CV_8U, cv::Scalar(0));
    frame.initialize_features(image, cameras, image);
    const Sophus::SE3f pose(Sophus::SO3f::exp(Eigen::Vector3f(0.1f, 0.2f, 0.3f)), Eigen::Vector3f(1, 2, 3));
    frame.set_pose(pose);
    ORB_SLAM3::map       map;
    ORB_SLAM3::key_frame keyframe(frame, &map, nullptr);
    cameras.clear();
    REQUIRE(keyframe.is_fisheye_stereo);
    REQUIRE(keyframe.descriptor(0).at<unsigned char>(0) == 1);
    REQUIRE(keyframe.descriptor(1).at<unsigned char>(0) == 2);
    REQUIRE(keyframe.get_features_in_area(35, 45, 1, true) == std::vector<std::size_t>{0});
    ORB_SLAM3::kannala_brandt8 camera(std::vector<float>{100, 100, 80, 80, 0, 0, 0, 0});
    const Sophus::SE3f         extrinsics(Sophus::SO3f::exp(Eigen::Vector3f(-0.1f, 0.1f, 0)), Eigen::Vector3f(0.2f, 0.01f, 0));
    camera.set_left_to_right_pose(extrinsics);
    const Sophus::SE3f right_pose = extrinsics.inverse() * pose;
    REQUIRE(keyframe.get_relative_pose_tlr(camera).matrix().isApprox(extrinsics.matrix()));
    REQUIRE(keyframe.get_relative_pose_trl(camera).matrix().isApprox(extrinsics.inverse().matrix()));
    REQUIRE(keyframe.get_right_pose(camera).matrix().isApprox(right_pose.matrix()));
    REQUIRE(keyframe.get_right_pose_inverse(camera).matrix().isApprox(right_pose.inverse().matrix()));
    REQUIRE(keyframe.get_right_camera_center(camera).isApprox(right_pose.inverse().translation()));
    REQUIRE(keyframe.get_right_rotation(camera).isApprox(right_pose.rotationMatrix()));
    REQUIRE(keyframe.get_right_translation(camera).isApprox(right_pose.translation()));

    const Eigen::Vector3f world_point = pose.inverse() * Eigen::Vector3f(0.2f, 0.3f, 2.f);
    ORB_SLAM3::map_point  point(world_point, &keyframe, &map);
    point.add_observation(&keyframe, 1);
    REQUIRE(point.get_observation_count() == 1);
    point.update_normal_and_depth(camera);
    REQUIRE(point.get_normal().isApprox((world_point - right_pose.inverse().translation()).normalized()));
}

TEST_CASE("Map point colors sample original pixels in RGB order", "[point_color]")
{
    ORB_SLAM3::frame frame;
    frame.keypoints                             = {cv::KeyPoint(1.f, 1.f, 1.f), cv::KeyPoint(-1.f, 0.f, 1.f), cv::KeyPoint(2.9f, 2.9f, 1.f)};
    frame.keypoints.front().pt_undistorted      = cv::Point2f(0.f, 0.f);
    const int channels                          = GENERATE(1, 3, 4);
    frame.color_image                           = cv::Mat(3, 3, CV_MAKETYPE(CV_8U, channels), cv::Scalar(10, 20, 30, 40));
    const std::array<unsigned char, 3> expected = channels == 1 ? std::array<unsigned char, 3>{10, 10, 10} : std::array<unsigned char, 3>{30, 20, 10};
    REQUIRE(frame.keypoint_color(0) == expected);
    REQUIRE(frame.keypoint_color(2) == expected);
    const std::array<unsigned char, 3> white{255, 255, 255};
    REQUIRE(frame.keypoint_color(-1) == white);
    REQUIRE(frame.keypoint_color(1) == white);
    REQUIRE(frame.keypoint_color(3) == white);
    frame.color_is_rgb                     = true;
    const std::array<unsigned char, 3> rgb = channels == 1 ? expected : std::array<unsigned char, 3>{10, 20, 30};
    REQUIRE(frame.keypoint_color(0) == rgb);
    frame.is_fisheye_stereo = true;
    frame.right_keypoints   = {cv::KeyPoint(0.f, 0.f, 1.f)};
    frame.color_image_right = cv::Mat(1, 1, CV_8UC1, cv::Scalar(70));
    const std::array<unsigned char, 3> gray{70, 70, 70};
    REQUIRE(frame.keypoint_color(3) == gray);
    frame.color_image.release();
    REQUIRE(frame.keypoint_color(0) == white);
}

TEST_CASE("Map points retain their first observation color", "[point_color]")
{
    cv::Mat image(128, 128, CV_8UC1);
    cv::RNG random(42);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::keypoint_extractor extractor(100, 1.2f, 1, 20, 7);
    ORB_SLAM3::camera_vector      cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::pinhole>(std::vector<float>{100, 100, 64, 64}));
    ORB_SLAM3::frame frame(0.0, nullptr, 40.f);
    ORB_SLAM3::tracking::extract_features(frame, extractor, image, cameras);
    REQUIRE(frame.feature_count >= 2);
    REQUIRE(frame.extraction_ms > 0);
    REQUIRE(frame.stereo_matching_ms >= 0);
    frame.optical_flow_ms = 2.0;
    const ORB_SLAM3::frame copied(frame);
    REQUIRE(copied.extraction_ms == frame.extraction_ms);
    REQUIRE(copied.stereo_matching_ms == frame.stereo_matching_ms);
    REQUIRE(copied.optical_flow_ms == frame.optical_flow_ms);
    const ORB_SLAM3::frame next(frame, 0.1, *cameras[0]);
    REQUIRE(next.extraction_ms == 0);
    REQUIRE(next.stereo_matching_ms == 0);
    REQUIRE(next.optical_flow_ms == 0);
    frame.set_pose(Sophus::SE3f{});
    frame.dataset     = 0;
    frame.color_image = cv::Mat(128, 128, CV_8UC3, cv::Scalar(10, 30, 200));
    ORB_SLAM3::map map;
    const auto     second_keypoint                                                          = frame.keypoints[1].pt;
    frame.color_image.at<cv::Vec3b>(cvRound(second_keypoint.y), cvRound(second_keypoint.x)) = cv::Vec3b(220, 40, 10);
    ORB_SLAM3::key_frame colored_keyframe(frame, &map, nullptr);
    REQUIRE_FALSE(colored_keyframe.is_fisheye_stereo);
    REQUIRE(colored_keyframe.keypoints[0].color == frame.keypoint_color(0));
    const std::array<unsigned char, 3> second_color{10, 40, 220};
    REQUIRE(colored_keyframe.keypoints[1].color == second_color);
    ORB_SLAM3::map_point               point;
    const std::array<unsigned char, 3> white{255, 255, 255};
    REQUIRE(point.color() == white);
    point.add_observation(&colored_keyframe, 0);
    REQUIRE(point.color() == colored_keyframe.keypoints[0].color);
    point.add_observation(&colored_keyframe, 1);
    REQUIRE(point.color() == colored_keyframe.keypoints[0].color);
}

TEST_CASE("Keypoints retain native detector metadata and separate tracking orientation", "[native_keypoints]")
{
    using detector  = ORB_SLAM3::keypoint_detector_type;
    const auto type = GENERATE(detector::fast,
                               detector::agast,
                               detector::gftt,
                               detector::orb,
                               detector::brisk,
                               detector::sift,
                               detector::akaze,
                               detector::star,
                               detector::harris_laplace,
                               detector::msd,
                               detector::tbmr,
                               detector::surf);
    CAPTURE(static_cast<int>(type));
#ifndef OPENCV_ENABLE_NONFREE
    if (type == detector::surf)
    {
        REQUIRE_THROWS_AS(ORB_SLAM3::keypoint_extractor(400, 1.2f, 3, 20, 7, type), cv::Exception);
        return;
    }
#endif
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 3, 20, 7, type);
    cv::Mat                       image(256, 256, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);
    std::vector<int>                 overlap{100, 200};
    std::vector<ORB_SLAM3::keypoint> tracking_keys;
    cv::Mat                          descriptors;
    const int                        mono_count = extractor(image, cv::noArray(), tracking_keys, descriptors, overlap);
    REQUIRE_FALSE(tracking_keys.empty());
    REQUIRE(descriptors.rows == static_cast<int>(tracking_keys.size()));
    REQUIRE(descriptors.cols == 32);
    REQUIRE(mono_count > 0);
    REQUIRE(mono_count < static_cast<int>(tracking_keys.size()));
    REQUIRE(std::any_of(tracking_keys.begin(), tracking_keys.end(), [](const auto &key) { return key.octave > 0; }));

    auto       raw_detector = reference_detector(type);
    const auto scales       = extractor.get_scale_factors();
    for (int level = 0; level < extractor.get_levels(); ++level)
    {
        std::vector<cv::KeyPoint> raw_keys;
        raw_detector->detect(extractor.image_pyramid[level], raw_keys);
        for (std::size_t index = 0; index < tracking_keys.size(); ++index)
        {
            const auto &tracking_key = tracking_keys[index];
            if (tracking_key.octave != level)
            {
                continue;
            }
            const auto &native = tracking_key;
            REQUIRE(tracking_key.tracking_angle >= 0);
            REQUIRE(tracking_key.tracking_angle < 360);
            const bool in_overlap = native.pt.x >= overlap[0] && native.pt.x <= overlap[1];
            REQUIRE(in_overlap == (index >= static_cast<std::size_t>(mono_count)));
            if (type == detector::fast || type == detector::agast)
            {
                REQUIRE(native.size == 7 * scales[level]);
                REQUIRE(native.angle == -1);
            }
            else
            {
                REQUIRE(std::any_of(raw_keys.begin(),
                                    raw_keys.end(),
                                    [&](const auto &raw)
                                    {
                                        return cv::norm(raw.pt * scales[level] - native.pt) < 0.001 && std::abs(raw.size * scales[level] - native.size) < 0.001 && raw.angle == native.angle &&
                                               raw.response == native.response && raw.class_id == native.class_id;
                                    }));
            }
        }
    }

    std::vector<ORB_SLAM3::keypoint> repeated_keys;
    cv::Mat                          repeated_descriptors;
    REQUIRE(extractor(image, cv::noArray(), repeated_keys, repeated_descriptors, overlap) == mono_count);
    REQUIRE(cv::norm(descriptors, repeated_descriptors, cv::NORM_INF) == 0);
    for (std::size_t index = 0; index < tracking_keys.size(); ++index)
    {
        REQUIRE(repeated_keys[index].pt == tracking_keys[index].pt);
        REQUIRE(repeated_keys[index].size == tracking_keys[index].size);
        REQUIRE(repeated_keys[index].angle == tracking_keys[index].angle);
        REQUIRE(repeated_keys[index].tracking_angle == tracking_keys[index].tracking_angle);
    }
    extractor(cv::Mat::zeros(image.size(), CV_8UC1), cv::noArray(), repeated_keys, repeated_descriptors, overlap);
    REQUIRE(repeated_keys.empty());
}

TEST_CASE("Keypoint extraction supports portrait images", "[native_keypoints]")
{
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 1, 20, 7);
    cv::Mat                       image(900, 320, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);

    std::vector<int>                 overlap{0, 0};
    std::vector<ORB_SLAM3::keypoint> keypoints;
    cv::Mat                          descriptors;

    extractor(image, cv::noArray(), keypoints, descriptors, overlap);

    REQUIRE_FALSE(keypoints.empty());
    REQUIRE(descriptors.rows == static_cast<int>(keypoints.size()));
    REQUIRE(descriptors.cols == 32);
}

TEST_CASE("OpenCV descriptors are selectable independently of the detector", "[native_keypoints][descriptors]")
{
    using namespace ORB_SLAM3;
    const auto detector   = GENERATE(keypoint_detector_type::fast, keypoint_detector_type::sift, keypoint_detector_type::brisk);
    const auto descriptor = GENERATE(keypoint_descriptor_type::orb,
                                     keypoint_descriptor_type::sift,
                                     keypoint_descriptor_type::brisk,
                                     keypoint_descriptor_type::brief,
                                     keypoint_descriptor_type::freak,
                                     keypoint_descriptor_type::latch,
                                     keypoint_descriptor_type::beblid,
                                     keypoint_descriptor_type::teblid,
                                     keypoint_descriptor_type::daisy,
                                     keypoint_descriptor_type::vgg,
                                     keypoint_descriptor_type::boostdesc,
                                     keypoint_descriptor_type::surf);
    CAPTURE(static_cast<int>(detector), static_cast<int>(descriptor));
#ifndef OPENCV_ENABLE_NONFREE
    if (descriptor == keypoint_descriptor_type::surf)
    {
        REQUIRE_THROWS_AS(keypoint_extractor(400, 1.2f, 3, 20, 7, detector, descriptor), cv::Exception);
        return;
    }
#endif
    keypoint_extractor extractor(400, 1.2f, 3, 20, 7, detector, descriptor);
    cv::Mat            image(256, 256, CV_8UC1);
    cv::RNG            random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);
    std::vector<int>      overlap{100, 200};
    std::vector<keypoint> keys;
    cv::Mat               descriptors;
    extractor(image, cv::noArray(), keys, descriptors, overlap);
    REQUIRE_FALSE(keys.empty());
    REQUIRE(descriptors.rows == static_cast<int>(keys.size()));
    const auto [columns, depth] = descriptor_layout(descriptor);
    REQUIRE(descriptors.type() == depth);
    REQUIRE(descriptors.cols == columns);
    REQUIRE(cv::checkRange(descriptors));
    REQUIRE(cv::norm(descriptor_matrix(keys), descriptors, cv::NORM_INF) == 0);
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        REQUIRE(keys[index].descriptor.type() == descriptors.type());
        REQUIRE(keys[index].descriptor.cols == descriptors.cols);
        if (depth == CV_32F && descriptor != keypoint_descriptor_type::sift)
        {
            REQUIRE(cv::norm(keys[index].descriptor, cv::NORM_L2) == Catch::Approx(512.0));
            REQUIRE(descriptor_matcher::descriptor_distance(keys[index].descriptor, -keys[index].descriptor) == Catch::Approx(256.0));
        }
    }
    std::vector<keypoint> repeated_keys;
    cv::Mat               repeated_descriptors;
    extractor(image, cv::noArray(), repeated_keys, repeated_descriptors, overlap);
    REQUIRE(cv::norm(descriptors, repeated_descriptors, cv::NORM_INF) == 0);
    REQUIRE(cv::norm(descriptor_matrix(repeated_keys), descriptors, cv::NORM_INF) == 0);
    extractor(cv::Mat(), cv::noArray(), keys, descriptors, overlap);
    REQUIRE(keys.empty());
    REQUIRE(descriptors.empty());
}

TEST_CASE("Keypoint extraction handles small pyramid levels", "[native_keypoints]")
{
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 2, 20, 7);
    cv::Mat                       image(70, 70, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);

    std::vector<int>                 overlap{0, 0};
    std::vector<ORB_SLAM3::keypoint> keypoints;
    cv::Mat                          descriptors;

    extractor(image, cv::noArray(), keypoints, descriptors, overlap);

    REQUIRE_FALSE(keypoints.empty());
    REQUIRE(descriptors.rows == static_cast<int>(keypoints.size()));
    REQUIRE(descriptors.cols == 32);
}

TEST_CASE("OpenCV matchers support binary and floating-point descriptors", "[descriptors][matching]")
{
    using namespace ORB_SLAM3;
    const auto backend    = GENERATE(descriptor_matcher_type::brute_force, descriptor_matcher_type::flann);
    const auto descriptor = GENERATE(keypoint_descriptor_type::orb,
                                     keypoint_descriptor_type::sift,
                                     keypoint_descriptor_type::brisk,
                                     keypoint_descriptor_type::brief,
                                     keypoint_descriptor_type::freak,
                                     keypoint_descriptor_type::latch,
                                     keypoint_descriptor_type::beblid,
                                     keypoint_descriptor_type::teblid,
                                     keypoint_descriptor_type::daisy,
                                     keypoint_descriptor_type::vgg,
                                     keypoint_descriptor_type::boostdesc,
                                     keypoint_descriptor_type::surf);
    CAPTURE(static_cast<int>(backend), static_cast<int>(descriptor));
    descriptor_matcher::set_backend(backend);
    struct reset_backend
    {
        ~reset_backend() { descriptor_matcher::set_backend(descriptor_matcher_type::brute_force); }
    } reset;
    const auto [columns, depth] = descriptor_layout(descriptor);
    cv::Mat train(100, columns, depth);
    cv::RNG random(71);
    random.fill(train, cv::RNG::UNIFORM, 0, 256);
    const auto                           query = train.row(37).clone();
    std::vector<std::vector<cv::DMatch>> matches;
    descriptor_matcher::knn_match(query, train, matches);
    REQUIRE(matches.size() == 1);
    REQUIRE_FALSE(matches.front().empty());
    REQUIRE(matches.front().front().trainIdx == 37);
    REQUIRE(matches.front().front().distance == 0);
    REQUIRE(descriptor_matcher::descriptor_distance(query, train.row(37)) == 0);
    const auto ranked = descriptor_matcher::rank_candidates(query, train, {12, 37, 45});
    REQUIRE_FALSE(ranked.empty());
    REQUIRE(ranked.front() == 37);
    descriptor_matcher::knn_match(query, train, matches, train.rows);
    REQUIRE(matches.size() == 1);
    REQUIRE(matches.front().size() == static_cast<std::size_t>(train.rows));
    REQUIRE(matches.front().front().trainIdx == 37);
    descriptor_matcher::knn_match(query, train.row(37), matches);
    REQUIRE(matches.size() == 1);
    REQUIRE(matches.front().size() == 1);
    descriptor_matcher::knn_match(query, cv::Mat(), matches);
    REQUIRE(matches.empty());
    REQUIRE(descriptor_matcher::rank_candidates(query, train, {}).empty());
}

TEST_CASE("Descriptor distances retain precision and account for binary length", "[descriptors][matching]")
{
    using ORB_SLAM3::descriptor_matcher;
    REQUIRE(descriptor_matcher::descriptor_distance(cv::Mat::zeros(1, 32, CV_8U), cv::Mat(1, 32, CV_8U, cv::Scalar(255))) == 256);
    REQUIRE(descriptor_matcher::descriptor_distance(cv::Mat::zeros(1, 64, CV_8U), cv::Mat(1, 64, CV_8U, cv::Scalar(255))) == 256);
    REQUIRE(descriptor_matcher::descriptor_distance(cv::Mat::zeros(1, 128, CV_32F), cv::Mat(1, 128, CV_32F, cv::Scalar(0.25))) > 0);
}
