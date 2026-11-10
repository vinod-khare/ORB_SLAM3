#include "MapPoint.h"
#include "Pinhole.h"
#include "Tracking.h"
#include "atlas.h"
#include "frame_drawer.h"
#include "keypoint_extractor.h"
#include "map_drawer.h"

#include <memory>
#include <type_traits>

#include <opencv2/imgproc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace
{

class frame_drawer_fixture : public ORB_SLAM3::frame_drawer
{
  public:
    frame_drawer_fixture(ORB_SLAM3::atlas &atlas, const std::vector<cv::KeyPoint> &keys, bool right, int image_type) : frame_drawer(atlas)
    {
        _state             = ORB_SLAM3::Tracking::OK;
        _localization_only = false;

        ORB_SLAM3::frame frame(0.0, nullptr, cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(4, 1, CV_32F), 10.f, 40.f);
        frame.color_image       = cv::Mat::zeros(160, 240, image_type);
        frame.color_image_right = frame.color_image.clone();
        frame.keypoints         = right ? std::vector<ORB_SLAM3::keypoint>(2) : std::vector<ORB_SLAM3::keypoint>{};
        frame.right_keypoints   = {};
        for (const auto &key : keys)
        {
            (right ? frame.right_keypoints : frame.keypoints).emplace_back(key);
        }
        const auto offset   = right ? frame.keypoints.size() : 0;
        frame.feature_count = static_cast<int>(offset + frame.right_keypoints.size());
        frame.map_points.assign(frame.feature_count, nullptr);
        frame.outliers.assign(frame.feature_count, false);
        _map_match.nObs              = 1;
        frame.map_points[offset]     = &_map_match;
        frame.map_points[offset + 1] = &_odometry_match;
        frame.map_points[offset + 3] = &_map_match;
        _current_frame               = std::make_shared<ORB_SLAM3::frame>(frame);
    }

    using frame_drawer::_current_frame;
    using frame_drawer::_tracked_count;
    using frame_drawer::_tracked_odometry_count;

  private:
    ORB_SLAM3::MapPoint _map_match;
    ORB_SLAM3::MapPoint _odometry_match;
};

} // namespace

TEST_CASE("Frame drawer renders tracked keypoint sizes and orientations in both camera views", "[frame_drawer]")
{
    const bool  right       = GENERATE(false, true);
    const float image_scale = GENERATE(0.5f, 1.f, 2.f);
    const int   image_type  = GENERATE(CV_8UC1, CV_8UC3);
    CAPTURE(right, image_scale, image_type);

    ORB_SLAM3::atlas                atlas(0);
    const std::vector<cv::KeyPoint> keys{
        cv::KeyPoint(40.f, 40.f, 24.f, 0.f), cv::KeyPoint(100.f, 40.f, 32.f, 90.f), cv::KeyPoint(160.f, 40.f, 20.f, 180.f), cv::KeyPoint(40.f, 100.f, 16.f, -1.f)};
    frame_drawer_fixture drawer(atlas, keys, right, image_type);

    cv::Mat expected    = cv::Mat::zeros(cv::Size(cvRound(240 / image_scale), cvRound(160 / image_scale)), CV_8UC3);
    auto    scaled_keys = keys;
    for (auto &key : scaled_keys)
    {
        key.pt   /= image_scale;
        key.size /= image_scale;
    }
    const auto flags = cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
    cv::drawKeypoints(expected, std::vector<cv::KeyPoint>{scaled_keys[0], scaled_keys[3]}, expected, cv::Scalar(0, 255, 0), flags);
    cv::drawKeypoints(expected, std::vector<cv::KeyPoint>{scaled_keys[1]}, expected, cv::Scalar(255, 0, 0), flags);

    for (int repeat = 0; repeat < 2; ++repeat)
    {
        const auto rendered = right ? drawer.draw_right_frame(image_scale) : drawer.draw_frame(image_scale);
        REQUIRE(rendered.cols == expected.cols);
        REQUIRE(rendered.rows > expected.rows);
        REQUIRE(rendered.type() == CV_8UC3);
        REQUIRE(cv::norm(rendered.rowRange(0, expected.rows), expected, cv::NORM_INF) == 0);
        REQUIRE(drawer._tracked_count == 2);
        REQUIRE(drawer._tracked_odometry_count == 1);
    }
}

static_assert(std::is_same_v<decltype(&ORB_SLAM3::frame_drawer::update), bool (ORB_SLAM3::frame_drawer::*)(ORB_SLAM3::Tracking &)>);

TEST_CASE("Off-screen map rendering survives pane resizing and navigation", "[.viewer_render]")
{
    ORB_SLAM3::atlas                 atlas(0);
    const ORB_SLAM3::viewer_settings settings{0.05f, 1.f, 1.f, 2.f, 0.08f, 3.f, 0.f, -0.7f, -1.8f, 500.f, 1.f};
    ORB_SLAM3::map_drawer            drawer(atlas, settings);
    const std::array<bool, 7>        layers{true, true, true, true, false, false, false};
    const auto                       first = drawer.render(640, 480, layers);
    REQUIRE(first.size() == cv::Size(640, 480));
    REQUIRE(first.type() == CV_8UC3);
    cv::Mat channels[3];
    cv::split(first, channels);
    REQUIRE(cv::countNonZero(channels[1] > channels[0] + 30) > 20);
    drawer.navigate(40, 20, 1, false);
    const auto moved = drawer.render(640, 480, layers);
    REQUIRE(cv::norm(first, moved, cv::NORM_L1) > 0);
    drawer.reset_view(true);
    drawer.follow_camera(true);
    const auto resized = drawer.render(320, 600, layers);
    REQUIRE(resized.size() == cv::Size(320, 600));
    drawer.release();
}

TEST_CASE("Map viewer renders sampled RGB for ordinary and reference points", "[.viewer_render][point_color_render]")
{
    ORB_SLAM3::atlas atlas(0);
    auto            &map = *atlas.get_current_map();
    cv::Mat          input(128, 128, CV_8UC1);
    cv::RNG          random(42);
    random.fill(input, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::keypoint_extractor extractor(100, 1.2f, 1, 20, 7);
    ORB_SLAM3::camera_vector      cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{100, 100, 64, 64}));
    cv::Mat distortion                     = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::frame::initial_computations = true;
    ORB_SLAM3::frame frame(0.0, nullptr, cameras[0]->toK(), distortion, 10.f, 40.f);
    ORB_SLAM3::Tracking::extract_features(frame, extractor, input, cameras);
    REQUIRE(frame.feature_count >= 2);
    frame.set_pose(Sophus::SE3f{});
    frame.dataset     = 0;
    frame.color_image = cv::Mat(128, 128, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int index = 0; index < 2; ++index)
    {
        const auto &position                                                      = frame.keypoints[index].pt;
        frame.color_image.at<cv::Vec3b>(cvRound(position.y), cvRound(position.x)) = index == 0 ? cv::Vec3b(0, 0, 255) : cv::Vec3b(255, 0, 0);
    }
    ORB_SLAM3::KeyFrame keyframe(frame, &map, nullptr, cameras);
    ORB_SLAM3::MapPoint red(Eigen::Vector3f(-0.2f, 0.f, 0.f), &keyframe, &map);
    ORB_SLAM3::MapPoint blue(Eigen::Vector3f(0.2f, 0.f, 0.f), &keyframe, &map);
    red.AddObservation(&keyframe, 0);
    blue.AddObservation(&keyframe, 1);
    map.add_map_point(&red);
    map.add_map_point(&blue);
    map.set_reference_map_points({&blue});
    const ORB_SLAM3::viewer_settings settings{0.05f, 1.f, 1.f, 12.f, 0.08f, 3.f, 0.f, 0.f, -1.8f, 500.f, 1.f};
    ORB_SLAM3::map_drawer            drawer(atlas, settings);
    const std::array<bool, 7>        layers{true, false, false, false, false, false, false};
    const auto                       image = drawer.render(640, 480, layers);
    cv::Mat                          red_mask;
    cv::Mat                          blue_mask;
    cv::inRange(image, cv::Scalar(240, 0, 0), cv::Scalar(255, 15, 15), red_mask);
    cv::inRange(image, cv::Scalar(0, 0, 240), cv::Scalar(15, 15, 255), blue_mask);
    REQUIRE(cv::countNonZero(red_mask) > 40);
    REQUIRE(cv::countNonZero(blue_mask) > 40);
    drawer.release();
}
