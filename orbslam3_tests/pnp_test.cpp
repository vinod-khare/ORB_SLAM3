#include "GeometricTools.h"
#include "KannalaBrandt8.h"
#include "Pinhole.h"

#include <algorithm>

#include <opencv2/calib3d.hpp>

#include <catch2/catch_test_macros.hpp>

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
    REQUIRE(ORB_SLAM3::GeometricTools::solve_pnp_ransac(world_points, image_points, max_errors, &camera, 20, estimated_pose, inliers));
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