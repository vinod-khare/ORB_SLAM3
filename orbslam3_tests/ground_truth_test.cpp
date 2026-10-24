#include "Frame.h"
#include "KeyFrame.h"
#include "MapDrawer.h"
#include "Pinhole.h"
#include "ground_truth_trajectory.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

namespace
{
struct ground_truth_file
{
    std::filesystem::path path = std::filesystem::temp_directory_path() / ("orbslam3_ground_truth_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~ground_truth_file() { std::filesystem::remove(path); }
    void write(const std::string &text) const { std::ofstream(path) << text; }
};
} // namespace

TEST_CASE("Ground truth reads TUM text and EuRoC CSV with camera offsets", "[ground_truth]")
{
    const ground_truth_file file;
    file.write("# timestamp tx ty tz qx qy qz qw\n\n  1\t0 0 0 0 0 0 1\n2 1 0 0 0 0 0 1\n3 0 1 0 0 0 0 1\n");
    const ORB_SLAM3::ground_truth_trajectory tum(file.path.string());
    REQUIRE(tum.samples().size() == 3);
    REQUIRE(tum.samples()[1].first == 2.0);
    REQUIRE(tum.samples()[1].second.isApprox(Eigen::Vector3d(1, 0, 0)));
    file.write("#timestamp [ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z\n1000000000,0,0,0,1,0,0,0,0,0,0\n2000000000,1,0,0,0,0,0,1,0,0,0\n");
    const ORB_SLAM3::ground_truth_trajectory csv(file.path.string(), Sophus::SE3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
    REQUIRE(csv.samples().size() == 2);
    REQUIRE(csv.samples()[1].first == 2.0);
    REQUIRE(csv.samples()[0].second.isApprox(Eigen::Vector3d(1, 0, 0)));
    REQUIRE(csv.samples()[1].second.isZero());
    const auto middle = csv.pose_at(1.5);
    REQUIRE(middle.has_value());
    REQUIRE(middle->translation().isApprox(Eigen::Vector3d(0.5, 1, 0)));
    REQUIRE(middle->rotationMatrix().isApprox(Eigen::AngleAxisd(std::acos(-1.0) / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix()));
    REQUIRE(csv.pose_at(1)->translation().isApprox(csv.samples()[0].second));
    REQUIRE(csv.pose_at(2)->translation().isApprox(csv.samples()[1].second));
    REQUIRE_FALSE(csv.pose_at(0.9));
    REQUIRE_FALSE(csv.pose_at(2.1));
    REQUIRE(ORB_SLAM3::ground_truth_trajectory("").samples().empty());
}

TEST_CASE("Ground truth rejects malformed poses and unordered timestamps", "[ground_truth]")
{
    const ground_truth_file file;
    for (const auto &text : {"", "1 0 0 0 0 0 0 1\n", "1 0 0\n", "1 0 0 0 0 0 0 0\n", "1 nan 0 0 0 0 0 1\n", "1 0 0 0 0 0 0 1 bad\n", "2 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n",
                             "1 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n", "1000000000,0,0,0,1,0,0\n", "1000000000,nan,0,0,1,0,0,0\n"})
    {
        CAPTURE(text);
        file.write(text);
        REQUIRE_THROWS(ORB_SLAM3::ground_truth_trajectory(file.path.string()));
    }
    REQUIRE_THROWS(ORB_SLAM3::ground_truth_trajectory(file.path.string() + ".missing"));
}

TEST_CASE("Ground truth aligns interpolated poses and optionally corrects scale", "[ground_truth]")
{
    const ground_truth_file file;
    file.write("1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 1\n3 0 1 0 0 0 0 1\n4 1 1 0 0 0 0 1\n");
    const ORB_SLAM3::ground_truth_trajectory trajectory(file.path.string());
    const Eigen::Matrix3d                    rotation = Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d                    translation(3, 4, 5);
    for (const double scale : {1.0, 2.5})
    {
        ORB_SLAM3::trajectory_samples estimate;
        for (const double timestamp : {1.2, 1.6, 2.3, 2.8, 3.7})
        {
            const auto pose = trajectory.pose_at(timestamp);
            REQUIRE(pose.has_value());
            estimate.emplace_back(timestamp, scale * rotation * pose->translation() + translation);
        }
        const auto alignment = trajectory.alignment(estimate, scale != 1.0);
        REQUIRE(alignment.has_value());
        REQUIRE(alignment->topLeftCorner<3, 3>().isApprox(scale * rotation));
        REQUIRE(alignment->topRightCorner<3, 1>().isApprox(translation));
        const auto rigid = trajectory.alignment(estimate, false);
        REQUIRE(rigid.has_value());
        REQUIRE(std::abs(rigid->topLeftCorner<3, 3>().determinant() - 1.0) < 1e-10);
    }
    REQUIRE_FALSE(trajectory.alignment({}, true));
    REQUIRE_FALSE(trajectory.alignment(
        {
            {10, Eigen::Vector3d::Zero()}
    },
        true));
    REQUIRE_FALSE(trajectory.alignment(
        {
            {    1, Eigen::Vector3d::Zero()},
            {1.001, Eigen::Vector3d::Zero()},
            {1.002, Eigen::Vector3d::Zero()}
    },
        true));
    file.write("1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 1\n3 2 0 0 0 0 0 1\n");
    const ORB_SLAM3::ground_truth_trajectory line(file.path.string());
    REQUIRE_FALSE(line.alignment(line.samples(), true));
}

TEST_CASE("Ground truth viewer settings use Kalibr camera-to-IMU calibration", "[ground_truth]")
{
    const ground_truth_file config;
    const ground_truth_file calibration;
    config.write("%YAML:1.0\n---\nViewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 1.0\nViewer.PointSize: 2.0\n"
                 "Viewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\nViewer.ViewpointX: 0.0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -1.8\nViewer.ViewpointF: 500.0\n"
                 "calibration:\n  file: " +
                 calibration.path.string() + "\n");
    calibration.write("cam0:\n  T_cam_imu: [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]\n");
    const auto settings = ORB_SLAM3::read_viewer_settings(config.path.string(), "ground_truth.csv", true, true);
    REQUIRE(settings.ground_truth_path == "ground_truth.csv");
    REQUIRE(settings.ground_truth_visible);
    REQUIRE(settings.ground_truth_correct_scale);
    REQUIRE(settings.ground_truth_body_to_camera.translation().isApprox(Eigen::Vector3d(-1, -2, -3)));
    calibration.write("cam0:\n  T_cam_imu: [[1, 0], [0, 1]]\n");
    REQUIRE_THROWS(ORB_SLAM3::read_viewer_settings(config.path.string(), "ground_truth.csv"));
    calibration.write("cam0:\n  T_cam_imu: [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n");
    REQUIRE_THROWS(ORB_SLAM3::read_viewer_settings(config.path.string(), "ground_truth.csv"));
    REQUIRE_NOTHROW(ORB_SLAM3::read_viewer_settings(config.path.string()));
}

TEST_CASE("Map view ground-truth layer renders and toggles independently", "[.viewer_render][ground_truth_render]")
{
    const ground_truth_file file;
    file.write("1 0 0 0 0 0 0 1\n2 0.3 0 0 0 0 0 1\n3 0.3 0.3 0 0 0 0 1\n4 0 0.3 0 0 0 0 1\n");
    ORB_SLAM3::viewer_settings settings{0.05f, 1.f, 1.f, 2.f, 0.08f, 3.f, 0.f, -0.7f, -1.8f, 500.f, 1.f};
    settings.ground_truth_path = file.path.string();
    ORB_SLAM3::Atlas              atlas(0);
    auto                         *map = atlas.GetCurrentMap();
    ORB_SLAM3::keypoint_extractor extractor(100, 1.2f, 1, 20, 7);
    ORB_SLAM3::Pinhole            camera({200, 200, 128, 128});
    cv::Mat                       image(256, 256, CV_8UC1);
    cv::RNG                       random(42);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::Mat distortion                      = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::Frame::mbInitialComputations = true;
    const ORB_SLAM3::ground_truth_trajectory          trajectory(file.path.string());
    std::vector<std::unique_ptr<ORB_SLAM3::KeyFrame>> keyframes;
    for (const double timestamp : {1.25, 2.25, 3.25})
    {
        ORB_SLAM3::Frame frame(image, timestamp, &extractor, nullptr, &camera, distortion, 0, 40);
        frame.mnDataset                = 0;
        const Eigen::Vector3f position = trajectory.pose_at(timestamp)->translation().cast<float>();
        frame.SetPose(Sophus::SE3f(Eigen::Quaternionf::Identity(), -position));
        keyframes.push_back(std::make_unique<ORB_SLAM3::KeyFrame>(frame, map, nullptr, &camera));
        map->AddKeyFrame(keyframes.back().get());
    }
    ORB_SLAM3::MapDrawer drawer(&atlas, settings);
    std::array<bool, 7>  layers{false, false, false, false, false, false, false};
    const auto           hidden = drawer.render(640, 480, layers);
    layers[5]                   = true;
    const auto visible          = drawer.render(640, 480, layers);
    cv::Mat    orange;
    cv::inRange(visible, cv::Scalar(220, 80, 0), cv::Scalar(255, 150, 40), orange);
    REQUIRE(cv::countNonZero(orange) > 50);
    REQUIRE(cv::norm(hidden, visible, cv::NORM_L1) > 0);
    for (const auto &keyframe : keyframes)
    {
        keyframe->SetPose(Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f(1, 2, 0)));
    }
    REQUIRE(cv::norm(visible, drawer.render(640, 480, layers), cv::NORM_INF) == 0);
    for (const auto &keyframe : keyframes)
    {
        map->EraseKeyFrame(keyframe.get());
    }
    REQUIRE(cv::norm(visible, drawer.render(640, 480, layers), cv::NORM_INF) == 0);
    for (const double timestamp : {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0})
    {
        drawer.SetCurrentCameraPose(trajectory.pose_at(timestamp)->cast<float>().inverse());
    }
    layers[6]       = true;
    const auto both = drawer.render(640, 480, layers);
    cv::Mat    teal;
    cv::inRange(both, cv::Scalar(0, 90, 140), cv::Scalar(30, 140, 190), teal);
    cv::inRange(both, cv::Scalar(220, 80, 0), cv::Scalar(255, 150, 40), orange);
    REQUIRE(cv::countNonZero(teal) > 50);
    REQUIRE(cv::countNonZero(orange) > 50);
    layers[6]            = false;
    layers[5]            = false;
    const auto no_trails = drawer.render(640, 480, layers);
    layers[6]            = true;
    REQUIRE(cv::norm(no_trails, drawer.render(640, 480, layers), cv::NORM_L1) > 0);
    drawer.release();
    for (const auto &keyframe : keyframes)
    {
        map->EraseKeyFrame(keyframe.get());
    }
}

TEST_CASE("Camera trail preserves hidden frames without bridging tracking loss or maps", "[.viewer_render][camera_trajectory_render]")
{
    const ORB_SLAM3::viewer_settings     settings{0.05f, 1.f, 1.f, 2.f, 0.08f, 3.f, 0.f, -0.7f, -1.8f, 500.f, 1.f};
    ORB_SLAM3::Atlas                     atlas(0);
    ORB_SLAM3::MapDrawer                 actual(&atlas, settings);
    ORB_SLAM3::MapDrawer                 expected(&atlas, settings);
    ORB_SLAM3::MapDrawer                 continuous(&atlas, settings);
    const std::array<Eigen::Vector3f, 6> positions{
        {{0, 0, 0}, {0.15f, 0, 0}, {0.35f, 0.15f, 0}, {0.5f, 0.15f, 0}, {0.7f, 0, 0}, {0.85f, 0, 0}}
    };
    for (std::size_t index = 0; index < positions.size(); ++index)
    {
        if (index == 2)
        {
            actual.break_camera_trajectory();
        }
        if (index % 2 == 0)
        {
            expected.break_camera_trajectory();
        }
        const Sophus::SE3f pose(Eigen::Quaternionf::Identity(), -positions[index]);
        actual.SetCurrentCameraPose(pose, index < 4 ? 0 : 1);
        expected.SetCurrentCameraPose(pose);
        continuous.SetCurrentCameraPose(pose);
    }
    std::array<bool, 7> layers{false, false, false, false, false, false, false};
    const auto          hidden = actual.render(640, 480, layers);
    layers[6]                  = true;
    const auto visible         = actual.render(640, 480, layers);
    REQUIRE(cv::norm(visible, expected.render(640, 480, layers), cv::NORM_INF) == 0);
    REQUIRE(cv::norm(visible, continuous.render(640, 480, layers), cv::NORM_L1) > 0);
    cv::Mat teal;
    cv::inRange(visible, cv::Scalar(0, 90, 140), cv::Scalar(30, 140, 190), teal);
    REQUIRE(cv::countNonZero(teal) > 50);
    layers[6] = false;
    REQUIRE(cv::norm(hidden, actual.render(640, 480, layers), cv::NORM_INF) == 0);
    actual.release();
    expected.release();
    continuous.release();
}