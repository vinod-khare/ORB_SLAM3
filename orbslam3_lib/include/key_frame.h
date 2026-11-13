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

#ifndef KEY_FRAME_H
#define KEY_FRAME_H

#include "frame.h"
#include "geometric_camera.h"
#include "imu_types.h"
#include "key_frame_database.h"
#include "keypoint_extractor.h"
#include "map_point.h"
#include "place_vocabulary.h"

#include <map>
#include <memory>
#include <mutex>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

class map;
class map_point;
class frame;
class key_frame_database;

class geometric_camera;

class key_frame
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    key_frame(frame &F, map *pMap, key_frame_database *pKFDB, const imu::calib &imu_calibration = {});

    // Pose functions
    void set_pose(const Sophus::SE3f &Tcw);
    void set_velocity(const Eigen::Vector3f &Vw_);

    Sophus::SE3f get_pose();

    Sophus::SE3f    get_pose_inverse();
    Eigen::Vector3f get_camera_center();

    Eigen::Vector3f get_imu_position();
    Eigen::Matrix3f get_imu_rotation();
    Sophus::SE3f    get_imu_pose();
    Eigen::Matrix3f get_rotation();
    Eigen::Vector3f get_translation();
    Eigen::Vector3f get_velocity();
    bool            is_velocity_set();

    // Bag of Words Representation
    void compute_bow();

    // Covisibility graph functions
    void add_connection(key_frame *pKF, const int &weight);
    void erase_connection(key_frame *pKF);

    void                     update_connections(bool upParent = true);
    void                     update_best_covisibles();
    std::set<key_frame *>    get_connected_key_frames();
    std::vector<key_frame *> get_vector_covisible_key_frames();
    std::vector<key_frame *> get_best_covisibility_key_frames(const int &number_of_key_frames);
    std::vector<key_frame *> get_covisibles_by_weight(const int &w);
    int                      get_weight(key_frame *pKF);

    // Spanning tree functions
    void                  add_child(key_frame *pKF);
    void                  erase_child(key_frame *pKF);
    void                  change_parent(key_frame *pKF);
    std::set<key_frame *> get_children();
    key_frame            *get_parent();
    bool                  has_child(key_frame *pKF);
    void                  set_first_connection(bool bFirst);

    // Loop Edges
    void                  add_loop_edge(key_frame *pKF);
    std::set<key_frame *> get_loop_edges();

    // Merge Edges
    void             add_merge_edge(key_frame *pKF);
    set<key_frame *> get_merge_edges();

    // map_point observation functions
    int                      get_map_point_count();
    void                     add_map_point(map_point *pMP, const size_t &idx);
    void                     erase_map_point_match(const int &idx);
    void                     erase_map_point_match(map_point *pMP);
    void                     replace_map_point_match(const int &idx, map_point *pMP);
    std::set<map_point *>    get_map_points();
    std::vector<map_point *> get_map_point_matches();
    int                      tracked_map_points(const int &minObs);
    map_point               *get_map_point(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> get_features_in_area(const float &x, const float &y, const float &r, const bool bRight = false) const;
    bool                unproject_stereo(int i, Eigen::Vector3f &x3D, geometric_camera &camera);

    // Image
    bool is_in_image(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void set_not_erase();
    void set_erase();

    // Set/check bad flag
    void set_bad_flag();
    bool is_bad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float compute_scene_median_depth(const int q);

    static bool compare_weight(int a, int b) { return a > b; }

    static bool compare_id(key_frame *pKF1, key_frame *pKF2) { return pKF1->id < pKF2->id; }

    map *get_map();
    void update_map(map *pMap);

    void            set_new_bias(const imu::bias &b);
    Eigen::Vector3f get_gyro_bias();

    Eigen::Vector3f get_accelerometer_bias();

    imu::bias get_imu_bias();

    bool project_point_distort(map_point *pMP, cv::Point2f &kp, float &u, float &v, geometric_camera &camera);
    bool project_point_undistort(map_point *pMP, cv::Point2f &kp, float &u, float &v, geometric_camera &camera);

    void set_key_frame_database(key_frame_database *pKFDB);

    bool has_imu;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
  public:
    static long unsigned int next_id;
    long unsigned int        id;
    const long unsigned int  frame_id;

    const double timestamp;

    // Grid (to speed up feature matching)
    const int   grid_columns;
    const int   grid_rows;
    const float grid_element_width_inverse;
    const float grid_element_height_inverse;

    // Variables used by the tracking
    long unsigned int track_reference_for_frame;
    long unsigned int fuse_target_key_frame_id;

    // Variables used by the local mapping
    long unsigned int local_ba_key_frame_id;
    long unsigned int fixed_ba_key_frame_id;

    // Number of optimizations by BA(amount of iterations in BA)
    long unsigned int number_of_optimizations;

    // Variables used by the keyframe database
    long unsigned int loop_query_id;
    int               loop_common_words;
    float             loop_score;
    long unsigned int relocalization_query_id;
    int               relocalization_words;
    float             relocalization_score;
    long unsigned int merge_query_id;
    int               merge_words;
    float             merge_score;
    long unsigned int place_recognition_query_id;
    int               place_recognition_words;
    float             place_recognition_score;

    bool current_place_recognition;

    // Variables used by loop closing
    Sophus::SE3f      camera_pose_global_ba;
    Sophus::SE3f      camera_pose_before_global_ba;
    Eigen::Vector3f   velocity_world_global_ba;
    Eigen::Vector3f   velocity_world_before_global_ba;
    imu::bias         bias_global_ba;
    long unsigned int global_ba_key_frame_id;

    // Variables used by merging
    Sophus::SE3f      camera_pose_merge;
    Sophus::SE3f      camera_pose_before_merge;
    Sophus::SE3f      world_pose_before_merge;
    Eigen::Vector3f   velocity_world_merge;
    Eigen::Vector3f   velocity_world_before_merge;
    imu::bias         bias_merge;
    long unsigned int merge_corrected_key_frame_id;
    long unsigned int merge_key_frame_id;
    float             scale_merge;
    long unsigned int local_ba_merge_key_frame_id;

    float scale;

    // Number of KeyPoints
    const int  feature_count;
    const bool is_fisheye_stereo;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<keypoint> keypoints;
    const std::vector<float>    right_coordinates; // negative value for monocular points
    const std::vector<float>    depths;            // negative value for monocular points
    const cv::Mat              &descriptor(std::size_t index) const
    {
        return is_fisheye_stereo && index >= keypoints.size() ? right_keypoints[index - keypoints.size()].descriptor : keypoints[index].descriptor;
    }

    // BoW
    DBoW2::BowVector     bow_vector;
    DBoW2::FeatureVector feature_vector;

    // Pose relative to parent (this is computed when bad flag is activated)
    Sophus::SE3f pose_relative_to_parent;

    // Scale
    const int                scale_levels;
    const float              scale_factor;
    const float              log_scale_factor;
    const std::vector<float> scale_factors;
    const std::vector<float> level_sigma_squared;
    const std::vector<float> inverse_level_sigma_squared;

    // Image bounds and calibration
    const int min_x;
    const int min_y;
    const int max_x;
    const int max_y;

    // Preintegrated IMU measurements from previous keyframe
    key_frame *previous_key_frame;
    key_frame *next_key_frame;

    imu::preintegrated *imu_preintegrated;
    imu::calib          imu_calibration;

    unsigned int origin_map_id;

    string name_file;

    int dataset;

    std::vector<key_frame *> loop_candidate_key_frames;
    std::vector<key_frame *> merge_candidate_key_frames;

    // The following variables need to be accessed trough a mutex to be thread safe.
  protected:
    // sophus poses
    Sophus::SE3<float> camera_pose;
    Eigen::Matrix3f    camera_rotation;
    Sophus::SE3<float> world_pose;
    Eigen::Matrix3f    world_rotation;

    // IMU position
    Eigen::Vector3f imu_position_world;
    // Velocity (Only used for inertial SLAM)
    Eigen::Vector3f velocity_world;
    bool            has_velocity;

    // Imu bias
    imu::bias imu_bias;

    // MapPoints associated to keypoints
    std::vector<map_point *> map_points;
    // For save relation without pointer, this is necessary for save/load function

    // BoW
    key_frame_database *database;
    place_vocabulary   *place_vocabulary_ptr;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> grid;

    std::map<key_frame *, int> connected_key_frame_weights;
    std::vector<key_frame *>   ordered_connected_key_frames;
    std::vector<int>           ordered_weights;
    // For save relation without pointer, this is necessary for save/load function

    // Spanning Tree and Loop Edges
    bool                  first_connection;
    key_frame            *parent_key_frame;
    std::set<key_frame *> child_key_frames;
    std::set<key_frame *> loop_edges;
    std::set<key_frame *> merge_edges;
    // For save relation without pointer, this is necessary for save/load function

    // Bad flags
    bool not_erasable;
    bool to_be_erased;
    bool bad;

    map *owner_map;

    // Mutex
    std::mutex pose_mutex; // for pose, velocity and biases
    std::mutex connections_mutex;
    std::mutex features_mutex;
    std::mutex map_mutex;

  public:
    // Indexes of stereo observations correspondences
    std::vector<int> left_to_right_matches, right_to_left_matches;

    Sophus::SE3f get_relative_pose_trl(const geometric_camera &camera);
    Sophus::SE3f get_relative_pose_tlr(const geometric_camera &camera);

    // KeyPoints in the right image (for stereo fisheye, coordinates are needed)
    const std::vector<keypoint> right_keypoints;

    std::vector<std::vector<std::vector<size_t>>> right_grid;

    Sophus::SE3<float> get_right_pose(const geometric_camera &camera);
    Sophus::SE3<float> get_right_pose_inverse(const geometric_camera &camera);

    Eigen::Vector3f            get_right_camera_center(const geometric_camera &camera);
    Eigen::Matrix<float, 3, 3> get_right_rotation(const geometric_camera &camera);
    Eigen::Vector3f            get_right_translation(const geometric_camera &camera);

    void print_point_distribution()
    {
        int       left = 0, right = 0;
        const int Nlim = is_fisheye_stereo ? static_cast<int>(keypoints.size()) : feature_count;
        for (int i = 0; i < feature_count; i++)
        {
            if (map_points[i])
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
        spdlog::debug("Point distribution in key_frame: left={}, right={}", left, right);
    }
};

} // namespace ORB_SLAM3

#endif // KEY_FRAME_H
