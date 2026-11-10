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

#ifndef SIM3_SOLVER_H
#define SIM3_SOLVER_H

#include "KeyFrame.h"

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

class sim3_solver
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    sim3_solver(KeyFrame                      *first_key_frame,
                KeyFrame                      *second_key_frame,
                const std::vector<MapPoint *> &matched_map_points,
                const bool                     fixed_scale        = true,
                const vector<KeyFrame *>       matched_key_frames = vector<KeyFrame *>());

    void set_ransac_parameters(double probability = 0.99, int minimum_inliers = 6, int maximum_iterations = 300);

    Eigen::Matrix4f find(std::vector<bool> &inliers_12, int &inlier_count);

    Eigen::Matrix4f iterate(int iteration_limit, bool &no_more_iterations, std::vector<bool> &inliers, int &inlier_count);
    Eigen::Matrix4f iterate(int iteration_limit, bool &no_more_iterations, vector<bool> &inliers, int &inlier_count, bool &converged);

    Eigen::Matrix4f get_estimated_transformation();
    Eigen::Matrix3f get_estimated_rotation();
    Eigen::Vector3f get_estimated_translation();
    float           get_estimated_scale();

  protected:
    void compute_centroid(Eigen::Matrix3f &points, Eigen::Matrix3f &centered_points, Eigen::Vector3f &centroid);

    bool compute_sim3(Eigen::Matrix3f &points_1, Eigen::Matrix3f &points_2);

    void check_inliers();

    void project(const std::vector<Eigen::Vector3f> &world_points, std::vector<Eigen::Vector2f> &image_points, Eigen::Matrix4f camera_from_world, GeometricCamera &camera);
    void from_camera_to_image(const std::vector<Eigen::Vector3f> &camera_points, std::vector<Eigen::Vector2f> &image_points, GeometricCamera &camera);

  protected:
    // KeyFrames and matches
    KeyFrame *key_frame_1;
    KeyFrame *key_frame_2;

    std::vector<Eigen::Vector3f> camera_points_1;
    std::vector<Eigen::Vector3f> camera_points_2;
    std::vector<MapPoint *>      map_points_1;
    std::vector<MapPoint *>      map_points_2;
    std::vector<MapPoint *>      matched_map_points_12;
    std::vector<size_t>          matched_indices_1;
    std::vector<size_t>          sigma_square_1;
    std::vector<size_t>          sigma_square_2;
    std::vector<size_t>          max_error_1;
    std::vector<size_t>          max_error_2;

    int number_of_correspondences;
    int initial_match_count;

    // Current Estimation
    Eigen::Matrix3f   rotation_12;
    Eigen::Vector3f   translation_12;
    float             scale_12;
    Eigen::Matrix4f   transformation_12;
    Eigen::Matrix4f   transformation_21;
    std::vector<bool> current_inliers;
    int               current_inlier_count;

    // Current Ransac State
    int               iterations;
    std::vector<bool> best_inliers;
    int               best_inlier_count;
    Eigen::Matrix4f   best_transformation_12;
    Eigen::Matrix3f   best_rotation;
    Eigen::Vector3f   best_translation;
    float             best_scale;

    // Scale is fixed to 1 in the stereo/RGBD case
    bool fix_scale;

    // Indices for random selection
    std::vector<size_t> all_indices;

    // Projections
    std::vector<Eigen::Vector2f> points_1_image_1;
    std::vector<Eigen::Vector2f> points_2_image_2;

    // RANSAC probability
    double ransac_probability;

    // RANSAC min inliers
    int ransac_min_inliers;

    // RANSAC max iterations
    int ransac_max_iterations;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*sigma_squared
    float threshold;
    float sigma_squared;

    // Calibration

    const camera_vector &cameras;
};

} // namespace ORB_SLAM3

#endif // SIM3_SOLVER_H
