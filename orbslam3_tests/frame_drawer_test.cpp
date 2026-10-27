#include "Atlas.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "MapPoint.h"
#include "Pinhole.h"
#include "Tracking.h"
#include "keypoint_extractor.h"

#include <chrono>
#include <future>

#include <opencv2/imgproc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace
{

class frame_drawer_fixture : public ORB_SLAM3::FrameDrawer
{
  public:
    frame_drawer_fixture(ORB_SLAM3::Atlas &atlas, const std::vector<cv::KeyPoint> &keys, bool right, int image_type) : FrameDrawer(atlas)
    {
        mState             = ORB_SLAM3::Tracking::OK;
        mbOnlyTracking     = false;
        mIm                = cv::Mat::zeros(160, 240, image_type);
        mImRight           = mIm.clone();
        mvCurrentKeys      = right ? std::vector<ORB_SLAM3::keypoint>(2) : std::vector<ORB_SLAM3::keypoint>{};
        mvCurrentKeysRight = right ? std::vector<ORB_SLAM3::keypoint>{} : std::vector<ORB_SLAM3::keypoint>{};
        for (const auto &key : keys)
        {
            (right ? mvCurrentKeysRight : mvCurrentKeys).emplace_back(key);
        }
        const auto offset = right ? mvCurrentKeys.size() : 0;
        mvbMap.assign(offset + keys.size(), false);
        mvbVO.assign(offset + keys.size(), false);
        mvbMap[offset]     = true;
        mvbVO[offset]      = true;
        mvbVO[offset + 1]  = true;
        mvbMap[offset + 3] = true;
    }

    using FrameDrawer::mMutex;
    using FrameDrawer::mnTracked;
    using FrameDrawer::mnTrackedVO;
};

} // namespace

TEST_CASE("Frame drawer renders tracked keypoint sizes and orientations in both camera views", "[frame_drawer]")
{
    const bool  right       = GENERATE(false, true);
    const float image_scale = GENERATE(0.5f, 1.f, 2.f);
    const int   image_type  = GENERATE(CV_8UC1, CV_8UC3);
    CAPTURE(right, image_scale, image_type);

    ORB_SLAM3::Atlas                atlas(0);
    const std::vector<cv::KeyPoint> keys{cv::KeyPoint(40.f, 40.f, 24.f, 0.f), cv::KeyPoint(100.f, 40.f, 32.f, 90.f), cv::KeyPoint(160.f, 40.f, 20.f, 180.f),
                                         cv::KeyPoint(40.f, 100.f, 16.f, -1.f)};
    frame_drawer_fixture            drawer(atlas, keys, right, image_type);

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
        const auto rendered = right ? drawer.DrawRightFrame(image_scale) : drawer.DrawFrame(image_scale);
        REQUIRE(rendered.cols == expected.cols);
        REQUIRE(rendered.rows > expected.rows);
        REQUIRE(rendered.type() == CV_8UC3);
        REQUIRE(cv::norm(rendered.rowRange(0, expected.rows), expected, cv::NORM_INF) == 0);
        REQUIRE(drawer.mnTracked == 2);
        REQUIRE(drawer.mnTrackedVO == 1);
    }
}

TEST_CASE("Camera publication drops frames instead of waiting for a busy UI", "[frame_drawer]")
{
    ORB_SLAM3::Atlas                atlas(0);
    const std::vector<cv::KeyPoint> keys{cv::KeyPoint(40, 40, 24), cv::KeyPoint(100, 40, 32), cv::KeyPoint(160, 40, 20), cv::KeyPoint(40, 100, 16)};
    frame_drawer_fixture            drawer(atlas, keys, false, CV_8UC3);
    std::unique_lock                lock(drawer.mMutex);
    auto                            publication = std::async(std::launch::async, [&] { return drawer.Update(nullptr); });
    const auto                      status      = publication.wait_for(std::chrono::milliseconds(200));
    lock.unlock();
    REQUIRE(status == std::future_status::ready);
    REQUIRE_FALSE(publication.get());
}

TEST_CASE("Off-screen map rendering survives pane resizing and navigation", "[.viewer_render]")
{
    ORB_SLAM3::Atlas                 atlas(0);
    const ORB_SLAM3::viewer_settings settings{0.05f, 1.f, 1.f, 2.f, 0.08f, 3.f, 0.f, -0.7f, -1.8f, 500.f, 1.f};
    ORB_SLAM3::MapDrawer             drawer(atlas, settings);
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
    ORB_SLAM3::Atlas atlas(0);
    auto            &map = *atlas.GetCurrentMap();
    cv::Mat          input(128, 128, CV_8UC1);
    cv::RNG          random(42);
    random.fill(input, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::keypoint_extractor extractor(100, 1.2f, 1, 20, 7);
    ORB_SLAM3::Pinhole            camera({100, 100, 64, 64});
    cv::Mat                       distortion = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::Frame::mbInitialComputations  = true;
    ORB_SLAM3::Frame frame(input, 0.0, &extractor, nullptr, &camera, distortion, 10.f, 40.f);
    REQUIRE(frame.N >= 2);
    frame.SetPose(Sophus::SE3f{});
    frame.mnDataset   = 0;
    frame.color_image = cv::Mat(128, 128, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int index = 0; index < 2; ++index)
    {
        const auto &position                                                      = frame.mvKeys[index].pt;
        frame.color_image.at<cv::Vec3b>(cvRound(position.y), cvRound(position.x)) = index == 0 ? cv::Vec3b(0, 0, 255) : cv::Vec3b(255, 0, 0);
    }
    ORB_SLAM3::KeyFrame keyframe(frame, &map, nullptr, &camera);
    ORB_SLAM3::MapPoint red(Eigen::Vector3f(-0.2f, 0.f, 0.f), &keyframe, &map);
    ORB_SLAM3::MapPoint blue(Eigen::Vector3f(0.2f, 0.f, 0.f), &keyframe, &map);
    red.AddObservation(&keyframe, 0);
    blue.AddObservation(&keyframe, 1);
    map.AddMapPoint(&red);
    map.AddMapPoint(&blue);
    map.SetReferenceMapPoints({&blue});
    const ORB_SLAM3::viewer_settings settings{0.05f, 1.f, 1.f, 12.f, 0.08f, 3.f, 0.f, 0.f, -1.8f, 500.f, 1.f};
    ORB_SLAM3::MapDrawer             drawer(atlas, settings);
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
