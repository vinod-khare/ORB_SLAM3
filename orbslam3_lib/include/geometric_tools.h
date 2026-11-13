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

#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

class key_frame;
class geometric_camera;

/** Shared camera geometry and matrix comparison helpers. */
class geometric_tools
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static bool solve_pnp_ransac(const std::vector<cv::Point3f> &world_points,
                                 const std::vector<cv::Point2f> &image_points,
                                 const std::vector<float>       &max_errors,
                                 geometric_camera               &camera,
                                 int                             min_inliers,
                                 Eigen::Matrix4f                &camera_from_world,
                                 std::vector<bool>              &inliers);

    // Compute the Fundamental matrix between KF1 and KF2
    static Eigen::Matrix3f compute_f12(key_frame *&key_frame1, key_frame *&key_frame2, geometric_camera &camera1, geometric_camera &camera2);

    // Triangulate point with KF1 and KF2
    static bool
    triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2, Eigen::Matrix<float, 3, 4> &camera1_from_world, Eigen::Matrix<float, 3, 4> &camera2_from_world, Eigen::Vector3f &point3d);

    template <int ROWS_T_, int COLS_T_> static bool check_matrices(const cv::Mat &cv_matrix, const Eigen::Matrix<float, ROWS_T_, COLS_T_> &eigen_matrix)
    {
        const float epsilon = 1e-3;
        if (ROWS_T_ != cv_matrix.rows || COLS_T_ != cv_matrix.cols)
        {
            spdlog::error("OpenCV matrix has the wrong dimensions");
            return false;
        }
        for (int i = 0; i < ROWS_T_; i++)
        {
            for (int j = 0; j < COLS_T_; j++)
            {
                if ((cv_matrix.at<float>(i, j) > (eigen_matrix(i, j) + epsilon)) || (cv_matrix.at<float>(i, j) < (eigen_matrix(i, j) - epsilon)))
                {
                    std::ostringstream matrix_output;
                    matrix_output << "OpenCV matrix:\n" << cv_matrix << "\nEigen matrix:\n" << eigen_matrix;
                    spdlog::debug("{}", matrix_output.str());
                    return false;
                }
            }
        }
        return true;
    }

    template <typename VALUE_T_, int ROWS_T_, int COLS_T_>
    static bool check_matrices(const Eigen::Matrix<VALUE_T_, ROWS_T_, COLS_T_> &eigen_matrix1, const Eigen::Matrix<VALUE_T_, ROWS_T_, COLS_T_> &eigen_matrix2)
    {
        const float epsilon = 1e-3;
        for (int i = 0; i < ROWS_T_; i++)
        {
            for (int j = 0; j < COLS_T_; j++)
            {
                if ((eigen_matrix1(i, j) > (eigen_matrix2(i, j) + epsilon)) || (eigen_matrix1(i, j) < (eigen_matrix2(i, j) - epsilon)))
                {
                    std::ostringstream matrix_output;
                    matrix_output << "Eigen matrix 1:\n" << eigen_matrix1 << "\nEigen matrix 2:\n" << eigen_matrix2;
                    spdlog::debug("{}", matrix_output.str());
                    return false;
                }
            }
        }
        return true;
    }
};

} // namespace ORB_SLAM3
