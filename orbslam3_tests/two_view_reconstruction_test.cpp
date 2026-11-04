#include "TwoViewReconstruction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEST_CASE("OpenCV two-view reconstruction recovers planar and nonplanar scenes", "[two_view]")
{
    const auto                            planar      = GENERATE(false, true);
    const auto                            noisy       = GENERATE(false, true);
    const Eigen::Matrix3f                 calibration = (Eigen::Matrix3f() << 500, 0, 320, 0, 510, 240, 0, 0, 1).finished();
    const Eigen::Matrix3f                 rotation    = Eigen::AngleAxisf(0.12f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    const Eigen::Vector3f                 translation(0.7f, -0.15f, 0.05f);
    ORB_SLAM3::TwoViewReconstruction      reconstruction(calibration);
    std::mt19937                          random(42);
    std::uniform_real_distribution<float> xy(planar ? -4.0f : -1.5f, planar ? 4.0f : 1.5f);
    std::uniform_real_distribution<float> depth(3.0f, 7.0f);
    std::normal_distribution<float>       noise(0, 0.25f);
    std::vector<cv::KeyPoint>             keys1;
    std::vector<cv::KeyPoint>             keys2(180);
    std::vector<Eigen::Vector3f>          world_points;
    std::vector<int>                      matches;
    for (int index = 0; index < 180; ++index)
    {
        const auto            x = xy(random);
        const auto            y = xy(random);
        const Eigen::Vector3f point(x, y, planar ? 4.5f + 0.4f * x + 0.2f * y : depth(random));
        const Eigen::Vector3f image1 = calibration * point;
        const Eigen::Vector3f image2 = calibration * (rotation * point + translation);
        keys1.emplace_back(cv::Point2f(image1.x() / image1.z(), image1.y() / image1.z()), 1);
        const auto target = 179 - index;
        keys2[target]     = cv::KeyPoint(cv::Point2f(image2.x() / image2.z() + noise(random) * noisy, image2.y() / image2.z() + noise(random) * noisy), 1);
        matches.push_back(index % 7 == 0 ? -1 : target);
        world_points.push_back(point);
        if (noisy && index < 25)
        {
            keys2[target].pt += cv::Point2f(75, -60);
        }
    }

    Sophus::SE3f             pose;
    std::vector<cv::Point3f> points;
    std::vector<bool>        triangulated;
    INFO("planar=" << planar << ", noisy=" << noisy);
    REQUIRE(reconstruction.Reconstruct(keys1, keys2, matches, pose, points, triangulated));
    REQUIRE(points.size() == keys1.size());
    REQUIRE(triangulated.size() == keys1.size());
    REQUIRE(std::abs(pose.translation().norm() - 1) < 1e-5f);
    REQUIRE((pose.rotationMatrix() - rotation).norm() < (noisy ? 0.03f : 1e-3f));
    REQUIRE((pose.translation() - translation.normalized()).norm() < (noisy ? 0.1f : 1e-3f));
    REQUIRE(std::count(triangulated.begin(), triangulated.end(), true) >= (noisy ? 115 : 150));
    for (std::size_t index = 0; index < matches.size(); ++index)
    {
        if (matches[index] < 0 || (noisy && index < 25))
        {
            REQUIRE_FALSE(triangulated[index]);
        }
        if (!triangulated[index])
        {
            continue;
        }
        const auto           &point = points[index];
        const Eigen::Vector3f point1(point.x, point.y, point.z);
        const Eigen::Vector3f point2 = pose * point1;
        REQUIRE(point1.z() > 0);
        REQUIRE(point2.z() > 0);
        const Eigen::Vector3f image1 = calibration * point1;
        const Eigen::Vector3f image2 = calibration * point2;
        const auto            error1 = cv::Point2f(image1.x() / image1.z(), image1.y() / image1.z()) - keys1[index].pt;
        const auto            error2 = cv::Point2f(image2.x() / image2.z(), image2.y() / image2.z()) - keys2[matches[index]].pt;
        REQUIRE(error1.dot(error1) <= 4.0f);
        REQUIRE(error2.dot(error2) <= 4.0f);
        if (!noisy)
        {
            REQUIRE((point1 - world_points[index] / translation.norm()).norm() < 1e-3f);
        }
    }
    // Reusing the initializer must not leave correspondence or output state behind.
    REQUIRE(reconstruction.Reconstruct(keys1, keys2, matches, pose, points, triangulated));
}

TEST_CASE("Two-view reconstruction rejects inadequate and degenerate geometry", "[two_view]")
{
    const Eigen::Matrix3f            calibration = (Eigen::Matrix3f() << 500, 0, 320, 0, 500, 240, 0, 0, 1).finished();
    ORB_SLAM3::TwoViewReconstruction reconstruction(calibration);
    Sophus::SE3f                     pose(Eigen::Matrix3f::Identity(), Eigen::Vector3f(1, 2, 3));
    std::vector<cv::Point3f>         points{cv::Point3f(1, 2, 3)};
    std::vector<bool>                triangulated{true};
    std::vector<cv::KeyPoint>        keys1;
    std::vector<cv::KeyPoint>        keys2;
    std::vector<int>                 matches;
    SECTION("Empty input") {}
    SECTION("Fewer than eight matches")
    {
        keys1.assign(7, cv::KeyPoint(cv::Point2f(320, 240), 1));
        keys2   = keys1;
        matches = {0, 1, 2, 3, 4, 5, 6};
    }
    SECTION("Coincident points")
    {
        keys1.assign(80, cv::KeyPoint(cv::Point2f(320, 240), 1));
        keys2 = keys1;
        matches.assign(80, 0);
    }
    SECTION("Collinear points")
    {
        for (int index = 0; index < 80; ++index)
        {
            keys1.emplace_back(cv::Point2f(index * 3.0f, 240), 1);
            keys2.emplace_back(cv::Point2f(index * 3.0f + 20, 240), 1);
            matches.push_back(index);
        }
    }
    SECTION("Ambiguous planar motion")
    {
        const Eigen::Matrix3f                 rotation = Eigen::AngleAxisf(0.12f, Eigen::Vector3f::UnitY()).toRotationMatrix();
        std::mt19937                          random(42);
        std::uniform_real_distribution<float> xy(-1.5f, 1.5f);
        for (int index = 0; index < 120; ++index)
        {
            const auto            x = xy(random);
            const auto            y = xy(random);
            const Eigen::Vector3f point(x, y, 4.5f + 0.8f * x + 0.4f * y);
            const Eigen::Vector3f image1 = calibration * point;
            const Eigen::Vector3f image2 = calibration * (rotation * point + Eigen::Vector3f(0.7f, -0.15f, 0.3f));
            keys1.emplace_back(cv::Point2f(image1.x() / image1.z(), image1.y() / image1.z()), 1);
            keys2.emplace_back(cv::Point2f(image2.x() / image2.z(), image2.y() / image2.z()), 1);
            matches.push_back(index);
        }
    }
    SECTION("Fewer than fifty supported points")
    {
        std::mt19937                          random(42);
        std::uniform_real_distribution<float> xy(-1.5f, 1.5f);
        std::uniform_real_distribution<float> depth(3.0f, 7.0f);
        for (int index = 0; index < 49; ++index)
        {
            const auto            x = xy(random);
            const auto            y = xy(random);
            const Eigen::Vector3f point(x, y, depth(random));
            const Eigen::Vector3f image1 = calibration * point;
            const Eigen::Vector3f image2 = calibration * (point + Eigen::Vector3f(0.7f, -0.15f, 0.05f));
            keys1.emplace_back(cv::Point2f(image1.x() / image1.z(), image1.y() / image1.z()), 1);
            keys2.emplace_back(cv::Point2f(image2.x() / image2.z(), image2.y() / image2.z()), 1);
            matches.push_back(index);
        }
    }
    SECTION("Pure rotation or low baseline")
    {
        const auto            baseline = GENERATE(0.0f, 1e-4f);
        const Eigen::Matrix3f rotation = Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitY()).toRotationMatrix();
        for (int index = 0; index < 120; ++index)
        {
            const Eigen::Vector3f point((index % 12 - 6) * 0.15f, (index / 12 - 5) * 0.15f, 4 + (index % 7) * 0.2f);
            const Eigen::Vector3f image1 = calibration * point;
            const Eigen::Vector3f image2 = calibration * (rotation * point + Eigen::Vector3f(baseline, 0, 0));
            keys1.emplace_back(cv::Point2f(image1.x() / image1.z(), image1.y() / image1.z()), 1);
            keys2.emplace_back(cv::Point2f(image2.x() / image2.z(), image2.y() / image2.z()), 1);
            matches.push_back(index);
        }
    }
    SECTION("Match list too long") { matches = {0}; }
    SECTION("Target index out of bounds")
    {
        keys1.emplace_back(cv::Point2f(320, 240), 1);
        matches = {0};
    }
    SECTION("Non-finite coordinates")
    {
        keys1.emplace_back(cv::Point2f(std::numeric_limits<float>::quiet_NaN(), 240), 1);
        keys2.emplace_back(cv::Point2f(320, 240), 1);
        matches = {0};
    }
    REQUIRE_FALSE(reconstruction.Reconstruct(keys1, keys2, matches, pose, points, triangulated));
    REQUIRE(pose.translation().isApprox(Eigen::Vector3f(1, 2, 3)));
    REQUIRE(points.size() == 1);
    REQUIRE(points[0] == cv::Point3f(1, 2, 3));
    REQUIRE(triangulated == std::vector<bool>{true});
}

TEST_CASE("Two-view reconstruction validates estimator parameters", "[two_view]")
{
    const Eigen::Matrix3f calibration = Eigen::Matrix3f::Identity();
    REQUIRE_THROWS_AS(ORB_SLAM3::TwoViewReconstruction(calibration, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(ORB_SLAM3::TwoViewReconstruction(calibration, 1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(ORB_SLAM3::TwoViewReconstruction(Eigen::Matrix3f::Zero()), std::invalid_argument);
}
