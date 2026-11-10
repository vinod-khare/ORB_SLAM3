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

#ifndef TRACKING_H
#define TRACKING_H

#include "GeometricCamera.h"
#include "ImuTypes.h"
#include "atlas.h"
#include "frame.h"
#include "key_frame_database.h"
#include "keypoint_extractor.h"
#include "local_mapping.h"
#include "loop_closing.h"
#include "place_vocabulary.h"
#include "settings.h"
#include "system.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace ORB_SLAM3
{

class local_mapping;
class loop_closing;
class system;
class settings;

class tracking
{

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    tracking(system              *pSys,
             place_vocabulary    *pVoc,
             atlas               *pAtlas,
             key_frame_database  *pKFDB,
             const int            sensor_type,
             settings            *settings,
             const camera_vector &camera_list,
             const string        &_nameSeq = std::string());

    ~tracking();

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f grab_image_stereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f grab_image_rgbd(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f grab_image_monocular(const cv::Mat &im, const double &timestamp, string filename);

    /** Extract current-image keypoints/descriptors and supply feature data to a frame. */
    static void
    extract_features(frame &frame, keypoint_extractor &extractor, const cv::Mat &left, const camera_vector &cameras, const cv::Mat &right = {}, keypoint_extractor *right_extractor = nullptr);

    void grab_imu_data(const IMU::Point &imuMeasurement);

    void set_local_mapper(local_mapping *pLocalMapper);
    void set_loop_closing(loop_closing *pLoopClosing);
    void set_reset_hooks(std::function<void()> on_reset_start, std::function<void()> on_reset_complete);
    void set_step_by_step(bool bSet);
    bool get_step_by_step();

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void inform_only_tracking(const bool &flag);

    void       update_frame_imu(const float s, const IMU::Bias &b, key_frame *pCurrentKeyFrame);
    key_frame *get_last_key_frame() { return last_key_frame; }

    void create_map_in_atlas();

    //--
    void new_dataset();
    int  get_number_dataset();
    int  get_matches_inliers();

    // DEBUG
    void save_sub_trajectory(string strNameFile_frames, string strNameFile_kf, string strFolder = "");
    void save_sub_trajectory(string strNameFile_frames, string strNameFile_kf, map *pMap);

    float get_image_scale();
    void  report_flow_statistics() const;

#ifdef REGISTER_LOOP
    void request_stop();
    bool is_stopped();
    void release();
    bool stop_requested();
#endif

  public:
    // Tracking states
    enum tracking_state
    {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET    = 0,
        NOT_INITIALIZED  = 1,
        OK               = 2,
        RECENTLY_LOST    = 3,
        LOST             = 4,
        OK_KLT           = 5
    };

    tracking_state state;
    tracking_state last_processed_state;

    // Input sensor
    int sensor;

    // Current Frame
    frame current_frame;
    frame last_frame;

    cv::Mat              gray_image;
    cv::Mat              color_image; // Original color image (BGR) for display
    cv::Mat              next_frame_color;
    cv::Mat              next_frame_color_right;
    std::vector<cv::Mat> next_optical_flow_pyramid;

    // Initialization Variables (Monocular)
    std::vector<int>         initial_last_matches;
    std::vector<int>         initial_matches;
    std::vector<cv::Point2f> previous_matched_keypoints;
    std::vector<cv::Point3f> initial_3d_points;
    frame                    initial_frame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> relative_frame_poses;
    list<key_frame *>  reference_key_frames;
    list<double>       frame_times;
    list<bool>         lost_frames;

    // frames with estimated pose
    int               tracked_frame_count;
    std::atomic<bool> step_by_step_enabled;

    // True if local mapping is deactivated and we are performing only localization
    bool only_tracking;

    void reset(bool bLocMap = false);
    void reset_active_map(bool bLocMap = false);

    float  mean_track;
    bool   initialize_with_three_key_frames;
    double first_frame_timestamp;        // time-stamp of first read frame
    double first_key_frame_timestamp;    // time-stamp of first inserted keyframe
    double imu_initialization_timestamp; // time-stamp of IMU initialization
    bool   fast_initialization = false;

    vector<map_point *> get_local_map_points();

    bool write_statistics;

  protected:
    void set_frame_colors(const cv::Mat &left, const cv::Mat &right = {});
    void create_frame(double timestamp, keypoint_extractor &extractor);

    // Main tracking function. It is independent of the input sensor.
    void track();

    // Map initialization for stereo and RGB-D
    void stereo_initialization();

    // Map initialization for monocular
    void monocular_initialization();
    void create_initial_map_monocular();

    void check_replaced_in_last_frame();
    bool track_reference_key_frame();
    void update_last_frame();
    bool track_with_motion_model();
    bool can_defer_features(double timestamp) const;
    void extract_current_features();
    int  track_with_optical_flow();
    bool predict_state_imu();

    bool relocalization();

    void update_local_map();
    void update_local_points();
    void update_local_key_frames();

    bool track_local_map();
    void search_local_points();

    bool need_new_key_frame();
    void create_new_key_frame();

    // Perform preintegration from last frame
    void preintegrate_imu();

    // Reset IMU biases and compute frame velocity
    void reset_frame_imu();

    bool map_updated;

    // Imu preintegration from last frame
    IMU::Preintegrated *imu_preintegrated_from_last_key_frame;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> imu_data_queue;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> imu_measurements_from_last_frame;
    std::mutex              imu_queue_mutex;

    // Imu calibration parameters
    IMU::Calib *imu_calibration;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias last_bias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool visual_odometry;

    // Other Thread Pointers
    local_mapping *local_mapper;
    loop_closing  *loop_closer;

    // ORB
    std::unique_ptr<keypoint_extractor> left_keypoint_extractor, right_keypoint_extractor;
    std::unique_ptr<keypoint_extractor> initial_keypoint_extractor;
    bool                                use_initial_extractor{false};

    // BoW
    place_vocabulary   *place_vocabulary_ptr;
    key_frame_database *database;

    // Initalization (only for monocular)
    bool ready_to_initialize;
    bool initialization_set;

    // Local Map
    key_frame               *reference_key_frame;
    std::vector<key_frame *> local_key_frames;
    std::vector<map_point *> local_map_points;

    // System
    system               *owner_system;
    std::function<void()> reset_start_callback;
    std::function<void()> reset_complete_callback;

    std::atomic<bool> step_by_step;

    // Atlas
    atlas *owner_atlas;

    // Calibration matrix
    cv::Mat         camera_matrix;
    Eigen::Matrix3f camera_matrix_eigen;
    cv::Mat         distortion_coefficients;
    float           baseline_fx;
    float           image_scale;

    float  imu_frequency;
    double imu_period;
    bool   insert_key_frames_when_lost;

    // New key_frame rules (according to fps)
    int minimum_frames;
    int maximum_frames;

    int first_imu_frame_id;
    int frames_to_reset_imu;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float depth_threshold;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float depth_map_factor;

    // Current matches in frame
    int matches_inliers;

    // Last Frame, key_frame and Relocalisation Info
    key_frame   *last_key_frame;
    unsigned int last_key_frame_id;
    unsigned int last_relocalization_frame_id;
    double       lost_timestamp;
    double       time_recently_lost;

    unsigned int first_frame_id;
    unsigned int initial_frame_id;
    unsigned int last_initialization_frame_id;

    bool map_created;

    // Motion Model
    bool         has_velocity{false};
    Sophus::SE3f velocity;

    bool          use_optical_flow{false};
    cv::Mat       previous_gray;
    cv::Mat       previous_right_gray;
    unsigned int  previous_gray_frame_id{0};
    unsigned long last_feature_frame_id{0};
    int           flow_reference_matches{0};
    std::size_t   flow_frames{0};
    std::size_t   feature_frames{0};
    std::size_t   flow_recoveries{0};

    // Color order (true RGB, false BGR, ignored if grayscale)
    bool rgb;

    list<map_point *> temporal_map_points;

    int dataset_count;

    ofstream f_track_stats;

    ofstream f_track_times;
    double   imu_preintegration_time;
    double   pose_prediction_time;
    double   local_map_tracking_time;
    double   new_key_frame_decision_time;

    const camera_vector &cameras;

    int initialization_id, last_id;

    Sophus::SE3f left_to_right_pose;

    void load_parameters(settings *settings);

#ifdef REGISTER_LOOP
    bool stop();

    bool       stopped;
    bool       stop_requested_flag;
    bool       not_stop;
    std::mutex stop_mutex;
#endif

  public:
    cv::Mat right_image;
};

} // namespace ORB_SLAM3

#endif // TRACKING_H
