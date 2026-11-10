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

#pragma once

#include "G2oCompat.h"

#include <vector>

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <sophus/geometry.hpp>
#include <sophus/sim3.hpp>

namespace ORB_SLAM3
{

/** Converts poses, rotations, and vectors between OpenCV, Eigen, g2o, and Sophus. */
class converter
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static g2o::SE3Quat to_se3_quat(const cv::Mat &transform);
    static g2o::SE3Quat to_se3_quat(const Sophus::SE3f &transform);
    static g2o::SE3Quat to_se3_quat(const g2o::Sim3 &similarity);

    static cv::Mat to_cv_mat(const g2o::SE3Quat &pose);
    static cv::Mat to_cv_mat(const g2o::Sim3 &similarity);

    template <typename DERIVED_T_> static cv::Mat to_cv_mat(const Eigen::MatrixBase<DERIVED_T_> &matrix)
    {
        cv::Mat cv_mat(static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols()), CV_32F);
        for (Eigen::Index row = 0; row < matrix.rows(); ++row)
        {
            for (Eigen::Index column = 0; column < matrix.cols(); ++column)
            {
                cv_mat.at<float>(static_cast<int>(row), static_cast<int>(column)) = static_cast<float>(matrix(row, column));
            }
        }

        return cv_mat;
    }

    static cv::Mat to_cv_se3(const Eigen::Matrix<double, 3, 3> &rotation, const Eigen::Matrix<double, 3, 1> &t);
    static cv::Mat to_cv_skew_matrix(const cv::Mat &v);

    static Eigen::Matrix<double, 3, 1> to_vector3d(const cv::Mat &vector);
    static Eigen::Matrix<float, 3, 1>  to_vector3f(const cv::Mat &vector);
    static Eigen::Matrix<double, 3, 1> to_vector3d(const cv::Point3f &point);
    static Eigen::Matrix<double, 3, 3> to_matrix3d(const cv::Mat &matrix);
    static Eigen::Matrix<double, 4, 4> to_matrix4d(const cv::Mat &matrix);
    static Eigen::Matrix<float, 3, 3>  to_matrix3f(const cv::Mat &matrix);
    static Eigen::Matrix<float, 4, 4>  to_matrix4f(const cv::Mat &matrix);
    static std::vector<float>          to_quaternion(const cv::Mat &matrix);

    static bool               is_rotation_matrix(const cv::Mat &rotation);
    static std::vector<float> to_euler(const cv::Mat &rotation);

    static Sophus::SE3<float> to_sophus(const cv::Mat &transform);
    static Sophus::Sim3f      to_sophus(const g2o::Sim3 &similarity);
};

} // namespace ORB_SLAM3
