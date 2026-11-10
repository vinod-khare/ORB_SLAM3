#include "G2oTypes.h"
#include "KannalaBrandt8.h"
#include "OptimizableTypes.h"
#include "Pinhole.h"
#include "frame.h"
#include "geometric_tools.h"

#include <algorithm>
#include <array>
#include <memory>
#include <sstream>
#include <type_traits>

#include <opencv2/calib3d.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

static_assert(std::has_virtual_destructor_v<ORB_SLAM3::GeometricCamera>);
static_assert(!std::is_copy_constructible_v<ORB_SLAM3::camera_vector::value_type>);

TEST_CASE("Camera collections distinguish rectified and fisheye stereo", "[camera_rig]")
{
    ORB_SLAM3::camera_vector cameras;
    REQUIRE_FALSE(ORB_SLAM3::is_fisheye_stereo(cameras));
    cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{500, 510, 320, 240}));
    REQUIRE_FALSE(ORB_SLAM3::is_fisheye_stereo(cameras));
    cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{500, 510, 320, 240}));
    REQUIRE_FALSE(ORB_SLAM3::is_fisheye_stereo(cameras));
    cameras.clear();
    cameras.push_back(std::make_unique<ORB_SLAM3::KannalaBrandt8>(std::vector<float>{500, 510, 320, 240, 0, 0, 0, 0}));
    REQUIRE_FALSE(ORB_SLAM3::is_fisheye_stereo(cameras));
    cameras.push_back(std::make_unique<ORB_SLAM3::KannalaBrandt8>(std::vector<float>{500, 510, 320, 240, 0, 0, 0, 0}));
    REQUIRE(ORB_SLAM3::is_fisheye_stereo(cameras));
}

namespace
{
template <typename Camera> void check_pnp_with_outliers(Camera &camera)
{
    std::vector<cv::Point3f> world_points;
    std::vector<cv::Point2f> image_points;
    std::vector<float>       max_errors;
    Eigen::Matrix3f          rotation = Eigen::AngleAxisf(0.08f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    const Eigen::Vector3f    translation(0.2f, -0.1f, 0.3f);

    for (int i = 0; i < 40; ++i)
    {
        const Eigen::Vector3f point((i % 5 - 2) * 0.35f, (i / 5 - 3) * 0.22f, 3.0f + (i % 7) * 0.25f);
        const Eigen::Vector3f camera_point = rotation * point + translation;
        const Eigen::Vector2f projected    = camera.project(camera_point);
        world_points.emplace_back(point.x(), point.y(), point.z());
        image_points.emplace_back(projected.x(), projected.y());
        max_errors.push_back(4.0f);
    }

    for (int i = 0; i < 10; ++i)
    {
        image_points[i].x += 80.0f;
        image_points[i].y -= 55.0f;
    }

    Eigen::Matrix4f   estimated_pose;
    std::vector<bool> inliers;
    REQUIRE(ORB_SLAM3::geometric_tools::solve_pnp_ransac(world_points, image_points, max_errors, camera, 20, estimated_pose, inliers));
    REQUIRE(std::count(inliers.begin(), inliers.end(), true) == 30);
    REQUIRE((estimated_pose.block<3, 3>(0, 0) - rotation).norm() < 1e-3f);
    REQUIRE((estimated_pose.block<3, 1>(0, 3) - translation).norm() < 1e-3f);
}
} // namespace

TEST_CASE("OpenCV PnP relocalization handles pinhole outliers")
{
    ORB_SLAM3::Pinhole camera({500.0f, 510.0f, 320.0f, 240.0f});
    check_pnp_with_outliers(camera);
}

TEST_CASE("OpenCV PnP relocalization handles Kannala-Brandt outliers")
{
    ORB_SLAM3::KannalaBrandt8 camera({500.0f, 510.0f, 320.0f, 240.0f, 0.01f, -0.001f, 0.0001f, -0.00001f});
    check_pnp_with_outliers(camera);
}

TEST_CASE("Geometric tools preserve triangulation and matrix comparison tolerances", "[geometry]")
{
    Eigen::Matrix<float, 3, 4> camera1 = Eigen::Matrix<float, 3, 4>::Zero();
    camera1.block<3, 3>(0, 0).setIdentity();
    auto camera2  = camera1;
    camera2(0, 3) = -1.f;
    Eigen::Vector3f ray1(0.f, 0.f, 1.f);
    Eigen::Vector3f ray2(-0.2f, 0.f, 1.f);
    Eigen::Vector3f point;
    REQUIRE(ORB_SLAM3::geometric_tools::triangulate(ray1, ray2, camera1, camera2, point));
    REQUIRE(point.isApprox(Eigen::Vector3f(0.f, 0.f, 5.f), 1e-5f));

    const Eigen::Matrix3f identity    = Eigen::Matrix3f::Identity();
    const cv::Mat         cv_identity = cv::Mat::eye(3, 3, CV_32F);
    REQUIRE(ORB_SLAM3::geometric_tools::check_matrices(cv_identity, identity));
    REQUIRE(ORB_SLAM3::geometric_tools::check_matrices(identity, identity));
    auto perturbed   = identity;
    perturbed(0, 0) += 0.0005f;
    REQUIRE(ORB_SLAM3::geometric_tools::check_matrices(identity, perturbed));
    perturbed(0, 0) += 0.001f;
    REQUIRE_FALSE(ORB_SLAM3::geometric_tools::check_matrices(identity, perturbed));
    REQUIRE_FALSE(ORB_SLAM3::geometric_tools::check_matrices(cv_identity, perturbed));
}

TEST_CASE("Optimizer edges bind their camera reference at construction", "[camera_rig]")
{
    static_assert(!std::is_default_constructible_v<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose>);
    static_assert(!std::is_default_constructible_v<ORB_SLAM3::EdgeSE3ProjectXYZ>);
    ORB_SLAM3::camera_vector cameras;
    cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{500, 510, 320, 240}));
    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose edge(*cameras[0]);
    REQUIRE(&edge.camera == cameras[0].get());
    const Eigen::Vector3f point(0, 0, 1);
    REQUIRE(edge.camera.project(point).isApprox(Eigen::Vector2f(320, 240)));
}

TEST_CASE("Inertial camera poses update only their modeled stereo observations", "[camera_rig]")
{
    const bool               fisheye = GENERATE(false, true);
    ORB_SLAM3::camera_vector cameras;
    for (int camera_index = 0; camera_index < 2; ++camera_index)
    {
        if (fisheye)
        {
            cameras.push_back(std::make_unique<ORB_SLAM3::KannalaBrandt8>(std::vector<float>{500, 510, 320, 240, 0, 0, 0, 0}));
        }
        else
        {
            cameras.push_back(std::make_unique<ORB_SLAM3::Pinhole>(std::vector<float>{500, 510, 320, 240}));
        }
    }
    ORB_SLAM3::frame frame;
    frame.set_pose(Sophus::SE3f{});
    frame.set_velocity(Eigen::Vector3f::Zero());
    frame.baseline_fx = 20;
    ORB_SLAM3::ImuCamPose       pose(&frame, cameras);
    const std::array<double, 6> update{};
    pose.Update(update.data());
    pose.UpdateW(update.data());
    REQUIRE(pose.cameras.data() == cameras.data());
    REQUIRE(pose.Rcw.size() == (fisheye ? 2 : 1));
    REQUIRE(pose.Project(Eigen::Vector3d(0, 0, 1)).isApprox(Eigen::Vector2d(320, 240)));
    static_assert(!std::is_default_constructible_v<ORB_SLAM3::VertexPose>);
    static_assert(!std::is_default_constructible_v<ORB_SLAM3::VertexPose4DoF>);
    ORB_SLAM3::VertexPose vertex(&frame, cameras);
    std::stringstream     serialized_pose;
    REQUIRE(vertex.write(serialized_pose));
    REQUIRE(vertex.read(serialized_pose));
    REQUIRE(vertex.estimate().cameras.data() == cameras.data());
    REQUIRE(vertex.estimate().Project(Eigen::Vector3d(0, 0, 1)).isApprox(Eigen::Vector2d(320, 240)));
}