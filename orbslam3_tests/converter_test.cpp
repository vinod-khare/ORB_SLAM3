#include "converter.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Converter preserves matrix dimensions and floating-point values", "[converter]")
{
    Eigen::Matrix<double, 2, 3> matrix;
    matrix << 1.25, -2.5, 3.75, 4.5, -5.25, 6.0;
    const auto converted = ORB_SLAM3::converter::to_cv_mat(matrix);
    REQUIRE(converted.rows == 2);
    REQUIRE(converted.cols == 3);
    REQUIRE(converted.type() == CV_32F);
    for (int row = 0; row < converted.rows; ++row)
    {
        for (int column = 0; column < converted.cols; ++column)
        {
            REQUIRE(converted.at<float>(row, column) == Catch::Approx(matrix(row, column)));
        }
    }

    const cv::Mat identity = cv::Mat::eye(4, 4, CV_32F);
    REQUIRE(ORB_SLAM3::converter::to_matrix3d(identity).isApprox(Eigen::Matrix3d::Identity()));
    REQUIRE(ORB_SLAM3::converter::to_matrix3f(identity).isApprox(Eigen::Matrix3f::Identity()));
    REQUIRE(ORB_SLAM3::converter::to_matrix4d(identity).isApprox(Eigen::Matrix4d::Identity()));
    REQUIRE(ORB_SLAM3::converter::to_matrix4f(identity).isApprox(Eigen::Matrix4f::Identity()));
}

TEST_CASE("Converter preserves rigid and scaled pose representations", "[converter]")
{
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d translation(1.0, -2.0, 3.0);
    const auto            matrix = ORB_SLAM3::converter::to_cv_se3(rotation, translation);
    const auto            pose   = ORB_SLAM3::converter::to_sophus(matrix);
    REQUIRE(pose.rotationMatrix().isApprox(rotation.cast<float>(), 1e-6f));
    REQUIRE(pose.translation().isApprox(translation.cast<float>(), 1e-6f));
    REQUIRE(cv::norm(ORB_SLAM3::converter::to_cv_mat(ORB_SLAM3::converter::to_se3_quat(matrix)), matrix, cv::NORM_INF) < 1e-6);
    REQUIRE(cv::norm(ORB_SLAM3::converter::to_cv_mat(ORB_SLAM3::converter::to_se3_quat(pose)), matrix, cv::NORM_INF) < 1e-6);

    const g2o::Sim3 similarity(rotation, translation, 2.0);
    const auto      scaled_matrix = ORB_SLAM3::converter::to_cv_mat(similarity);
    REQUIRE(ORB_SLAM3::converter::to_matrix3d(scaled_matrix).isApprox(2.0 * rotation, 1e-6));
    REQUIRE(ORB_SLAM3::converter::to_vector3d(scaled_matrix.col(3)).isApprox(translation, 1e-6));
    const auto scaled_pose = ORB_SLAM3::converter::to_sophus(similarity);
    REQUIRE(scaled_pose.scale() == Catch::Approx(2.0));
    REQUIRE(scaled_pose.translation().isApprox(translation.cast<float>(), 1e-6f));
}

TEST_CASE("Converter preserves vector and rotation representations", "[converter]")
{
    const cv::Mat vector = cv::Mat(cv::Vec3f(1.f, -2.f, 3.f));
    REQUIRE(ORB_SLAM3::converter::to_vector3d(vector).isApprox(Eigen::Vector3d(1, -2, 3)));
    REQUIRE(ORB_SLAM3::converter::to_vector3f(vector).isApprox(Eigen::Vector3f(1, -2, 3)));
    REQUIRE(ORB_SLAM3::converter::to_vector3d(cv::Point3f(1, -2, 3)).isApprox(Eigen::Vector3d(1, -2, 3)));
    const auto skew = ORB_SLAM3::converter::to_cv_skew_matrix(vector);
    REQUIRE(cv::norm(skew * vector, cv::NORM_INF) == 0);

    const cv::Mat identity = cv::Mat::eye(3, 3, CV_32F);
    REQUIRE(ORB_SLAM3::converter::is_rotation_matrix(identity));
    REQUIRE(ORB_SLAM3::converter::to_quaternion(identity) == std::vector<float>{0.f, 0.f, 0.f, 1.f});
    REQUIRE(ORB_SLAM3::converter::to_euler(identity) == std::vector<float>{0.f, 0.f, 0.f});
}
