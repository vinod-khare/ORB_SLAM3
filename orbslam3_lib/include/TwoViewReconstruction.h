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

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3
{

/** OpenCV two-view geometry with SLAM-specific model and motion acceptance checks. */
class TwoViewReconstruction
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TwoViewReconstruction(const Eigen::Matrix3f &k, float sigma = 1.0, int iterations = 200);

    /**
     * Recover a unit-baseline pose and points indexed by the first view's keypoints.
     * @return False for insufficient, degenerate, or ambiguous geometry; outputs are unchanged on failure.
     */
    bool Reconstruct(const std::vector<cv::KeyPoint> &keys1,
                     const std::vector<cv::KeyPoint> &keys2,
                     const std::vector<int>          &matches12,
                     Sophus::SE3f                    &pose21,
                     std::vector<cv::Point3f>        &points3d,
                     std::vector<bool>               &triangulated);

  private:
    bool reconstruct_model(
        const cv::Mat &model, bool use_homography, const std::vector<bool> &inliers, Sophus::SE3f &pose21, std::vector<cv::Point3f> &points3d, std::vector<bool> &triangulated) const;
    float score_homography(const cv::Mat &homography, std::vector<bool> &inliers) const;
    float score_fundamental(const cv::Mat &fundamental, std::vector<bool> &inliers) const;

    int check_motion(const Eigen::Matrix3f    &rotation,
                     const Eigen::Vector3f    &translation,
                     const std::vector<bool>  &inliers,
                     std::vector<cv::Point3f> &points3d,
                     std::vector<bool>        &triangulated,
                     float                    &parallax) const;

    cv::Mat                  calibration_;
    float                    sigma_;
    int                      max_iterations_;
    std::size_t              reference_size_{};
    std::vector<cv::Point2f> matched_points1_;
    std::vector<cv::Point2f> matched_points2_;
    std::vector<std::size_t> reference_indices_;
};

} // namespace ORB_SLAM3
