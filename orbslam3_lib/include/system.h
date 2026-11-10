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

#ifndef SYSTEM_H
#define SYSTEM_H

#include "CameraModels/GeometricCamera.h"
#include "ImuTypes.h"
#include "atlas.h"
#include "feature_overrides.h"
#include "keypoint.h"
#include "processing_statistics.h"

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{

using std::string;
using std::vector;

class key_frame_database;
class map;
class map_point;
class place_vocabulary;
class tracking;
class local_mapping;
class loop_closing;
class settings;

class system
{
  public:
    // Input sensor
    enum sensor_type
    {
        monocular     = 0,
        stereo        = 1,
        rgbd          = 2,
        imu_monocular = 3,
        imu_stereo    = 4,
        imu_rgbd      = 5,
    };

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Initialize the SLAM system. It launches the Local Mapping and Loop Closing threads.
    system(const string            &strVocFile,
           const string            &strSettingsFile,
           const sensor_type            sensor,
           const int                initFr      = 0,
           const string            &strSequence = std::string(),
           const feature_overrides &overrides   = {});
    ~system();

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f track_stereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point> &vImuMeas = vector<IMU::Point>(), string filename = "");

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f track_rgbd(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point> &vImuMeas = vector<IMU::Point>(), string filename = "");

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f track_monocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point> &vImuMeas = vector<IMU::Point>(), string filename = "");

    /** Supply original BGR images for display and map colors on the next tracking call. */
    void set_next_frame_color(const cv::Mat &color_image, const cv::Mat &color_image_right = {});
    bool can_precompute_optical_flow_pyramid() const;
    void set_next_optical_flow_pyramid(std::vector<cv::Mat> pyramid);
    int  dbow_version() const;
    void set_dbow_version(int version);

    // This stops local mapping thread (map building) and performs only camera tracking.
    void activate_localization_mode();
    // This resumes local mapping thread and performs SLAM again.
    void deactivate_localization_mode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool map_changed();

    // reset the system (clear Atlas or the active map)
    void reset();
    void reset_active_map();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void shutdown();
    bool is_shut_down();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void save_trajectory_tum(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void save_key_frame_trajectory_tum(const string &filename);

    void save_trajectory_euroc(const string &filename);
    void save_key_frame_trajectory_euroc(const string &filename);

    void save_trajectory_euroc(const string &filename, map *pMap);
    void save_key_frame_trajectory_euroc(const string &filename, map *pMap);

    // Export a COLMAP-compatible sparse model (cameras.txt, images.txt, points3D.txt)
    // plus a debug CSV (poses_debug.csv) into strDir.
    // images.txt uses world-to-camera Tcw convention with quaternion order qw qx qy qz.
    // Call after shutdown().
    void save_colmap(const string &strDir);

    // Export map points to ASCII PLY point cloud file(s).
    // If multiple maps exist, files are suffixed with _map<id>.
    // Example filename "pointcloud.ply" -> pointcloud_map0.ply, pointcloud_map1.ply
    // Call after shutdown().
    void save_point_cloud_ply(const string &filename);

    // Save data used for initialization debug
    void save_debug_data(const int &iniIdx);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void save_trajectory_kitti(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after track_monocular (or stereo or rgbd)
    int                      get_tracking_state();
    std::vector<map_point *> get_tracked_map_points();
    std::vector<keypoint>    get_tracked_keypoints_undistorted();

    // For debugging
    double get_time_from_imu_init();
    bool   is_lost();
    bool   is_finished();

    void change_dataset();

    float                get_image_scale();
    atlas               *get_atlas() const;
    tracking            *get_tracker() const;
    float                get_camera_frame_rate() const;
    const camera_vector &cameras() const { return _cameras; }

    processing_statistics       &statistics() { return _statistics; }
    const processing_statistics &statistics() const { return _statistics; }
    void                         record_frame_statistics(double preprocessing_ms, double frontend_ms);

  private:
    void stop_workers();

    // Keep filename history aligned with tracker frame-pose history in System,
    // so Tracking remains unchanged.
    void register_frame_filename_for_export(const string &filename);

    // Input sensor
    sensor_type _sensor;

    std::unique_ptr<settings> _settings;
    camera_vector             _cameras;
    processing_statistics     _statistics;

    // ORB vocabulary used for place recognition and feature matching.
    std::unique_ptr<place_vocabulary> _vocabulary;

    // key_frame database for place recognition (relocalization and loop detection).
    std::unique_ptr<key_frame_database> _key_frame_database;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    // Map* mpMap;
    std::unique_ptr<atlas> _atlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    std::unique_ptr<tracking> _tracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    std::unique_ptr<local_mapping> _local_mapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    std::unique_ptr<loop_closing> _loop_closer;

    // System threads: Local Mapping and Loop Closing.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::mutex   _shutdown_mutex;
    std::jthread _local_mapping_thread;
    std::jthread _loop_closing_thread;

    // reset flag
    std::mutex _reset_mutex;
    bool       _reset_requested;
    bool       _active_map_reset_requested;

    // Change mode flags
    std::mutex _mode_mutex;
    bool       _localization_mode_activation_requested;
    bool       _localization_mode_deactivation_requested;

    // shutdown flag
    bool _shutdown;

    // Tracking state
    int                      _tracking_state;
    std::vector<map_point *> _tracked_map_points;
    std::vector<keypoint>    _tracked_keypoints_undistorted;
    std::mutex               _state_mutex;

    // One filename per tracked frame pose entry (same cardinality as
    // _tracker->relative_frame_poses when exported).
    std::list<string> _frame_filenames{};

    //
    string _vocabulary_file_path;
};

} // namespace ORB_SLAM3

#endif // SYSTEM_H
