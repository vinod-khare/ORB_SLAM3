/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "converter.h"

#include <cassert>
#include <cmath>

namespace ORB_SLAM3
{

g2o::SE3Quat converter::to_se3_quat(const cv::Mat &transform)
{
    Eigen::Matrix<double, 3, 3> rotation;
    rotation << transform.at<float>(0, 0), transform.at<float>(0, 1), transform.at<float>(0, 2), transform.at<float>(1, 0), transform.at<float>(1, 1), transform.at<float>(1, 2),
        transform.at<float>(2, 0), transform.at<float>(2, 1), transform.at<float>(2, 2);

    Eigen::Matrix<double, 3, 1> t(transform.at<float>(0, 3), transform.at<float>(1, 3), transform.at<float>(2, 3));

    return g2o::SE3Quat(rotation, t);
}

g2o::SE3Quat converter::to_se3_quat(const Sophus::SE3f &transform) { return g2o::SE3Quat(transform.unit_quaternion().cast<double>(), transform.translation().cast<double>()); }

cv::Mat converter::to_cv_mat(const g2o::SE3Quat &pose)
{
    Eigen::Matrix<double, 4, 4> eigen_matrix = pose.to_homogeneous_matrix();
    return to_cv_mat(eigen_matrix);
}

cv::Mat converter::to_cv_mat(const g2o::Sim3 &similarity)
{
    Eigen::Matrix3d rotation    = similarity.rotation().toRotationMatrix();
    Eigen::Vector3d translation = similarity.translation();
    double          s           = similarity.scale();
    return to_cv_se3(s * rotation, translation);
}

cv::Mat converter::to_cv_se3(const Eigen::Matrix<double, 3, 3> &rotation, const Eigen::Matrix<double, 3, 1> &t)
{
    cv::Mat matrix = cv::Mat::eye(4, 4, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            matrix.at<float>(i, j) = rotation(i, j);
        }
    }
    for (int i = 0; i < 3; i++)
    {
        matrix.at<float>(i, 3) = t(i);
    }

    return matrix.clone();
}

Eigen::Matrix<double, 3, 1> converter::to_vector3d(const cv::Mat &vector)
{
    Eigen::Matrix<double, 3, 1> v;
    v << vector.at<float>(0), vector.at<float>(1), vector.at<float>(2);

    return v;
}

Eigen::Matrix<float, 3, 1> converter::to_vector3f(const cv::Mat &vector)
{
    Eigen::Matrix<float, 3, 1> v;
    v << vector.at<float>(0), vector.at<float>(1), vector.at<float>(2);

    return v;
}

Eigen::Matrix<double, 3, 1> converter::to_vector3d(const cv::Point3f &point)
{
    Eigen::Matrix<double, 3, 1> v;
    v << point.x, point.y, point.z;

    return v;
}

Eigen::Matrix<double, 3, 3> converter::to_matrix3d(const cv::Mat &matrix)
{
    Eigen::Matrix<double, 3, 3> eigen_matrix;

    eigen_matrix << matrix.at<float>(0, 0), matrix.at<float>(0, 1), matrix.at<float>(0, 2), matrix.at<float>(1, 0), matrix.at<float>(1, 1), matrix.at<float>(1, 2), matrix.at<float>(2, 0),
        matrix.at<float>(2, 1), matrix.at<float>(2, 2);

    return eigen_matrix;
}

Eigen::Matrix<double, 4, 4> converter::to_matrix4d(const cv::Mat &matrix)
{
    Eigen::Matrix<double, 4, 4> eigen_matrix;

    eigen_matrix << matrix.at<float>(0, 0), matrix.at<float>(0, 1), matrix.at<float>(0, 2), matrix.at<float>(0, 3), matrix.at<float>(1, 0), matrix.at<float>(1, 1), matrix.at<float>(1, 2),
        matrix.at<float>(1, 3), matrix.at<float>(2, 0), matrix.at<float>(2, 1), matrix.at<float>(2, 2), matrix.at<float>(2, 3), matrix.at<float>(3, 0), matrix.at<float>(3, 1),
        matrix.at<float>(3, 2), matrix.at<float>(3, 3);
    return eigen_matrix;
}

Eigen::Matrix<float, 3, 3> converter::to_matrix3f(const cv::Mat &matrix)
{
    Eigen::Matrix<float, 3, 3> eigen_matrix;

    eigen_matrix << matrix.at<float>(0, 0), matrix.at<float>(0, 1), matrix.at<float>(0, 2), matrix.at<float>(1, 0), matrix.at<float>(1, 1), matrix.at<float>(1, 2), matrix.at<float>(2, 0),
        matrix.at<float>(2, 1), matrix.at<float>(2, 2);

    return eigen_matrix;
}

Eigen::Matrix<float, 4, 4> converter::to_matrix4f(const cv::Mat &matrix)
{
    Eigen::Matrix<float, 4, 4> eigen_matrix;

    eigen_matrix << matrix.at<float>(0, 0), matrix.at<float>(0, 1), matrix.at<float>(0, 2), matrix.at<float>(0, 3), matrix.at<float>(1, 0), matrix.at<float>(1, 1), matrix.at<float>(1, 2),
        matrix.at<float>(1, 3), matrix.at<float>(2, 0), matrix.at<float>(2, 1), matrix.at<float>(2, 2), matrix.at<float>(2, 3), matrix.at<float>(3, 0), matrix.at<float>(3, 1),
        matrix.at<float>(3, 2), matrix.at<float>(3, 3);
    return eigen_matrix;
}

std::vector<float> converter::to_quaternion(const cv::Mat &matrix)
{
    Eigen::Matrix<double, 3, 3> eigen_matrix = to_matrix3d(matrix);
    Eigen::Quaterniond          q(eigen_matrix);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

cv::Mat converter::to_cv_skew_matrix(const cv::Mat &v)
{
    return cv::Mat(cv::Matx33f(0.f, -v.at<float>(2), v.at<float>(1), v.at<float>(2), 0.f, -v.at<float>(0), -v.at<float>(1), v.at<float>(0), 0.f));
}

bool converter::is_rotation_matrix(const cv::Mat &rotation)
{
    cv::Mat transpose;
    cv::transpose(rotation, transpose);
    cv::Mat should_be_identity = transpose * rotation;
    cv::Mat identity           = cv::Mat::eye(3, 3, should_be_identity.type());

    return cv::norm(identity, should_be_identity) < 1e-6;
}

std::vector<float> converter::to_euler(const cv::Mat &rotation)
{
    assert(is_rotation_matrix(rotation));
    float sy = sqrt(rotation.at<float>(0, 0) * rotation.at<float>(0, 0) + rotation.at<float>(1, 0) * rotation.at<float>(1, 0));

    bool singular = sy < 1e-6;

    float x;
    float y;
    float z;
    if (!singular)
    {
        x = atan2(rotation.at<float>(2, 1), rotation.at<float>(2, 2));
        y = atan2(-rotation.at<float>(2, 0), sy);
        z = atan2(rotation.at<float>(1, 0), rotation.at<float>(0, 0));
    }
    else
    {
        x = atan2(-rotation.at<float>(1, 2), rotation.at<float>(1, 1));
        y = atan2(-rotation.at<float>(2, 0), sy);
        z = 0;
    }

    std::vector<float> v_euler(3);
    v_euler[0] = x;
    v_euler[1] = y;
    v_euler[2] = z;

    return v_euler;
}

Sophus::SE3<float> converter::to_sophus(const cv::Mat &transform)
{
    Eigen::Matrix<double, 3, 3> eigen_matrix = to_matrix3d(transform.rowRange(0, 3).colRange(0, 3));
    Eigen::Quaternionf          q(eigen_matrix.cast<float>());

    Eigen::Matrix<float, 3, 1> t = to_vector3d(transform.rowRange(0, 3).col(3)).cast<float>();

    return Sophus::SE3<float>(q, t);
}

Sophus::Sim3f converter::to_sophus(const g2o::Sim3 &similarity)
{
    return Sophus::Sim3f(Sophus::RxSO3d((float)similarity.scale(), similarity.rotation().matrix()).cast<float>(), similarity.translation().cast<float>());
}

} // namespace ORB_SLAM3
