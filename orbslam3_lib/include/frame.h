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

#include "Eigen/Core"
#include "converter.h"
#include "imu_types.h"
#include "keypoint.h"
#include "place_vocabulary.h"
#include "settings.h"
#include "sophus/se3.hpp"

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <sophus/geometry.hpp>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class map_point;
class key_frame;
class constraint_pose_imu;
class geometric_camera;

class frame
{
  public:
    frame();

    std::array<unsigned char, 3> keypoint_color(int index) const;
    cv::Mat                      color_image;
    cv::Mat                      color_image_right;
    bool                         color_is_rgb{false};

    // Copy constructor.
    frame(const frame &frame);

    // Calibration/state for a new image, without detecting or describing features.
    frame(frame &previous, double timestamp);
    /** Initialize geometry and feature indexing from keypoints supplied by Tracking. */
    void initialize_features(
        const cv::Mat &left, const camera_vector &cameras, const cv::Mat &right = {}, const std::vector<cv::Mat> &left_pyramid = {}, const std::vector<cv::Mat> &right_pyramid = {});
    void set_tracked_features(std::vector<keypoint> keys, std::vector<map_point *> points, std::vector<keypoint> right_keys = {}, std::vector<map_point *> right_points = {});
    bool reliable_flow_inliers(const cv::Size &image_size, int minimum_inliers = 50);
    bool flow_only{false};
    int  flow_candidates{};
    int  flow_tracks{};
    int  flow_inliers{};
    int  flow_occupied_cells{};

    /** Construct calibration/state only; Tracking supplies extracted features separately. */
    frame(double              timestamp,
          place_vocabulary   *vocabulary,
          const cv::Mat      &calibration,
          const cv::Mat      &distortion,
          float               baseline_fx,
          float               depth_threshold,
          frame              *previous        = nullptr,
          const imu::calib   &imu_calibration = {},
          const Sophus::SE3f &left_to_right   = {});

    const std::vector<cv::Mat> &get_optical_flow_pyramid(const cv::Mat &image, bool right = false);
    static std::vector<cv::Mat> build_optical_flow_pyramid(const cv::Mat &image);

    // Compute Bag of Words representation.
    void compute_bow();

    // Set the camera pose. (Imu pose is not modified!)
    void set_pose(const Sophus::SE3<float> &Tcw);

    // Set IMU velocity
    void set_velocity(Eigen::Vector3f Vw);

    Eigen::Vector3f get_velocity() const;

    // Set IMU pose and velocity (implicitly changes camera pose)
    void set_imu_pose_velocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);

    Eigen::Matrix<float, 3, 1> get_imu_position() const;
    Eigen::Matrix<float, 3, 3> get_imu_rotation();
    Sophus::SE3<float>         get_imu_pose();

    Sophus::SE3f    get_relative_pose_trl();
    Sophus::SE3f    get_relative_pose_tlr();
    Eigen::Matrix3f get_relative_pose_tlr_rotation();
    Eigen::Vector3f get_relative_pose_tlr_translation();

    void set_new_bias(const imu::bias &b);

    // Check if a map_point is in the frustum of the camera
    // and fill variables of the map_point to be used by the tracking
    bool is_in_frustum(map_point *pMP, float viewingCosLimit, const camera_vector &cameras);

    bool project_point_distort(map_point *pMP, cv::Point2f &kp, float &u, float &v);

    Eigen::Vector3f in_reference_coordinates(Eigen::Vector3f pCw);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool position_in_grid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> get_features_in_area(const float &x, const float &y, const float &r, const int minLevel = -1, const int maxLevel = -1, const bool bRight = false) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void compute_stereo_matches(const std::vector<cv::Mat> &left_pyramid, const std::vector<cv::Mat> &right_pyramid);

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void compute_stereo_from_rgbd(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    bool unproject_stereo(const int &i, Eigen::Vector3f &x3D);

    constraint_pose_imu *imu_constraint;

    bool imu_is_preintegrated() const;
    void set_integrated();

    bool is_set() const;
    // Computes rotation, translation and camera center matrices from the camera pose.
    void update_pose_matrices();

    // Returns the camera center.
    inline Eigen::Vector3f get_camera_center() { return _camera_origin; }

    // Returns inverse of rotation
    inline Eigen::Matrix3f get_rotation_inverse() { return _world_from_camera_rotation; }

    inline Sophus::SE3<float> get_pose() const
    {
        // TODO: can the Frame pose be accsessed from several threads? should this be protected somehow?
        return _camera_from_world;
    }

    inline Eigen::Matrix3f get_rwc() const { return _world_from_camera_rotation; }

    inline Eigen::Vector3f get_camera_origin() const { return _camera_origin; }

    inline bool has_pose() const { return _has_pose; }

    inline bool has_velocity() const { return _has_velocity; }

  private:
    // Sophus/Eigen migration
    Sophus::SE3<float>         _camera_from_world;
    Eigen::Matrix<float, 3, 3> _world_from_camera_rotation;
    Eigen::Matrix<float, 3, 1> _camera_origin;
    Eigen::Matrix<float, 3, 3> _world_to_camera_rotation;
    Eigen::Matrix<float, 3, 1> _world_to_camera_translation;
    bool                       _has_pose;

    // Rcw_ not necessary as Sophus has a method for extracting the rotation matrix: Tcw_.rotationMatrix()
    // tcw_ not necessary as Sophus has a method for extracting the translation vector: Tcw_.translation()
    // Twc_ not necessary as Sophus has a method for easily computing the inverse pose: Tcw_.inverse()

    Sophus::SE3<float>         _left_to_right, _right_to_left;
    Eigen::Matrix<float, 3, 3> _left_to_right_rotation;
    Eigen::Vector3f            _left_to_right_translation;

    // IMU linear velocity
    Eigen::Vector3f _world_velocity;
    bool            _has_velocity;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Vocabulary used for relocalization.
    place_vocabulary *place_vocabulary_ptr;

    // Frame timestamp.
    double timestamp;

    std::vector<cv::Mat> optical_flow_pyramid;
    std::vector<cv::Mat> right_optical_flow_pyramid;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat         calibration_matrix;
    Eigen::Matrix3f eigen_calibration_matrix;
    static float    fx;
    static float    fy;
    static float    cx;
    static float    cy;
    static float    invfx;
    static float    invfy;
    cv::Mat         distortion_coefficients;

    // Stereo baseline multiplied by fx.
    float baseline_fx;

    // Stereo baseline in meters.
    float baseline;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float depth_threshold;

    // Number of KeyPoints.
    int feature_count;

    // Keypoints retain original image coordinates in pt and geometry coordinates in pt_undistorted.
    double                extraction_ms{};
    double                stereo_matching_ms{};
    double                optical_flow_ms{};
    std::vector<keypoint> keypoints, right_keypoints;
    bool                  is_fisheye_stereo{false};

    // Corresponding stereo coordinate and depth for each keypoint.
    std::vector<map_point *> map_points;
    // "Monocular" keypoints have a negative value.
    std::vector<float> right_coordinates;
    std::vector<float> depths;

    // Bag of Words Vector structures.
    DBoW2::BowVector     bow_vector;
    DBoW2::FeatureVector feature_vector;

    const cv::Mat &descriptor(std::size_t index) const
    {
        return is_fisheye_stereo && index >= keypoints.size() ? right_keypoints[index - keypoints.size()].descriptor : keypoints[index].descriptor;
    }

    // MapPoints associated to keypoints, NULL pointer if no association.
    // Flag to identify outlier associations.
    std::vector<bool> outliers;
    int               close_map_point_count;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float             grid_element_width_inverse;
    static float             grid_element_height_inverse;
    std::vector<std::size_t> grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    imu::bias predicted_imu_bias;

    // IMU bias
    imu::bias imu_bias;

    // Imu calibration
    imu::calib imu_calibration;

    // Imu preintegration from last keyframe
    imu::preintegrated *imu_preintegrated;
    key_frame          *last_key_frame;

    // Pointer to previous frame
    frame              *previous_frame;
    imu::preintegrated *frame_imu_preintegrated;

    // Current and Next Frame id.
    static long unsigned int next_id;
    long unsigned int        id;

    // Reference Keyframe.
    key_frame *reference_key_frame;

    // Scale pyramid info.
    int           scale_levels;
    float         scale_factor;
    float         log_scale_factor;
    vector<float> scale_factors;
    vector<float> inverse_scale_factors;
    vector<float> level_sigma_squared;
    vector<float> inverse_level_sigma_squared;

    // Undistorted Image Bounds (computed once).
    static float min_x;
    static float max_x;
    static float min_y;
    static float max_y;

    static bool initial_computations;

    std::map<long unsigned int, cv::Point2f> projected_points;
    std::map<long unsigned int, cv::Point2f> matched_in_image;

    string name_file;

    int dataset;

  private:
    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void _undistort_keypoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void _compute_image_bounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void _assign_features_to_grid();

    bool _is_set;

    bool _imu_preintegrated;

    std::shared_ptr<std::mutex> _imu_mutex{std::make_shared<std::mutex>()};

  public:
    // Number of Non Lapping Keypoints
    int mono_left, mono_right;

    // For stereo matching
    std::vector<int> left_to_right_matches, right_to_left_matches;

    // For stereo fisheye matching

    // Triangulated stereo observations using as reference the left camera. These are
    // computed during compute_stereo_fisheye_matches
    std::vector<Eigen::Vector3f> stereo_3d_points;

    // Grid for the right image
    std::vector<std::size_t> right_grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Stereo fisheye
    void compute_stereo_fisheye_matches(const camera_vector &cameras);

    bool is_in_frustum_checks(map_point *pMP, float viewingCosLimit, geometric_camera &camera, bool bRight = false);

    Eigen::Vector3f unproject_stereo_fisheye(const int &i);

    cv::Mat left_image, right_image;

    void print_point_distribution()
    {
        int       left = 0, right = 0;
        const int Nlim = is_fisheye_stereo ? static_cast<int>(keypoints.size()) : feature_count;
        for (int i = 0; i < feature_count; i++)
        {
            if (map_points[i] && !outliers[i])
            {
                if (i < Nlim)
                {
                    left++;
                }
                else
                {
                    right++;
                }
            }
        }
        spdlog::debug("Point distribution in Frame: left={}, right={}", left, right);
    }

    Sophus::SE3<double> test_pose;
};

} // namespace ORB_SLAM3
