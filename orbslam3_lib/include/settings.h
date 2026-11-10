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

#ifndef ORB_SLAM3_SETTINGS_H
#define ORB_SLAM3_SETTINGS_H

#include "CameraModels/GeometricCamera.h"
#include "descriptor_matcher_type.h"
#include "feature_overrides.h"
#include "keypoint_descriptor_type.h"
#include "keypoint_detector_type.h"

#include <filesystem>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

class system;

class settings
{
  public:
    /*
     * Enum for the different camera types implemented
     */
    enum camera_model_type
    {
        pinhole        = 0,
        kannala_brandt = 2
    };

    /*
     * Delete default constructor
     */
    settings() = delete;

    /*
     * Constructor from file
     */
    settings(const std::string &config_file, const int &sensor, const feature_overrides &overrides = {});

    /*
     * Ostream operator overloading to dump settings to the terminal
     */
    friend std::ostream &operator<<(std::ostream &output, const settings &s);

    /*
     * Getter methods
     */
    camera_model_type    camera_type() { return _camera_type; }
    const camera_vector &cameras() const { return _cameras; }
    cv::Mat              camera_1_distortion_coefficients() { return cv::Mat(_camera_1_distortion.size(), 1, CV_32F, _camera_1_distortion.data()); }
    cv::Mat              camera_2_distortion_coefficients() { return cv::Mat(_camera_2_distortion.size(), 1, CV_32F, _camera_2_distortion.data()); }

    Sophus::SE3f left_to_right_pose() { return _left_to_right_pose; }
    float        baseline_fx() { return _baseline_fx; }
    float        baseline() { return _baseline; }
    float        depth_threshold() { return _depth_threshold; }

    bool need_to_undistort() { return _need_to_undistort; }

    cv::Size new_image_size() { return _new_image_size; }
    float    fps() { return _fps; }
    bool     rgb() { return _rgb; }
    bool     need_to_resize() { return _need_to_resize_left; }
    bool     need_to_rectify() { return _need_to_rectify; }

    float        gyroscope_noise() { return _gyroscope_noise; }
    float        accelerometer_noise() { return _accelerometer_noise; }
    float        gyroscope_random_walk() { return _gyroscope_random_walk; }
    float        accelerometer_random_walk() { return _accelerometer_random_walk; }
    float        imu_frequency() { return _imu_frequency; }
    Sophus::SE3f imu_to_camera_pose() { return _imu_to_camera_pose; }
    bool         insert_key_frames_when_lost() { return _insert_key_frames_when_lost; }

    float depth_map_factor() { return _depth_map_factor; }

    int                      feature_count() { return _feature_count; }
    int                      level_count() { return _level_count; }
    float                    initial_fast_threshold() { return _initial_fast_threshold; }
    float                    minimum_fast_threshold() { return _minimum_fast_threshold; }
    float                    scale_factor() { return _scale_factor; }
    keypoint_detector_type   keypoint_detector() const { return _keypoint_detector; }
    keypoint_descriptor_type keypoint_descriptor() const { return _keypoint_descriptor; }
    descriptor_matcher_type  matcher_backend() const { return _descriptor_matcher; }
    bool                     use_optical_flow() const { return _use_optical_flow; }
    int                      dbow_version() const { return _dbow_version; }

    float far_point_threshold() { return _far_point_threshold; }

    cv::Mat left_rectification_map_x() { return _left_rectification_map_x; }
    cv::Mat left_rectification_map_y() { return _left_rectification_map_y; }
    cv::Mat right_rectification_map_x() { return _right_rectification_map_x; }
    cv::Mat right_rectification_map_y() { return _right_rectification_map_y; }

  private:
    friend class system;

    template <typename T> T read_parameter(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required = true)
    {
        cv::FileNode node = fSettings[name];
        if (node.empty())
        {
            if (required)
            {
                spdlog::error("{} required parameter does not exist, aborting...", name);
                exit(-1);
            }
            else
            {
                spdlog::debug("{} optional parameter does not exist...", name);
                found = false;
                return T();
            }
        }
        else
        {
            found = true;
            return (T)node;
        }
    }

    void read_cameras(cv::FileStorage &settings);
    void read_image_info(cv::FileStorage &fSettings);
    void read_imu(cv::FileStorage &settings, const std::string &config_file);
    void read_rgbd(cv::FileStorage &fSettings);
    void read_features(cv::FileStorage &fSettings, const std::string &config_file, const feature_overrides &overrides);
    void read_other_parameters(cv::FileStorage &fSettings);

    void precompute_rectification_maps();

    int                   _sensor;
    std::filesystem::path _calibration_file; // Kalibr camchain file
    camera_model_type     _camera_type;      // Camera type

    /*
     * Visual stuff
     */
    camera_vector      _cameras;
    camera_vector      _original_cameras;
    std::vector<float> _camera_1_distortion, _camera_2_distortion;

    cv::Size _original_image_size, _new_image_size;
    float    _fps;
    bool     _rgb;

    bool _need_to_undistort;
    bool _need_to_rectify;
    bool _need_to_resize_left, _need_to_resize_right;

    Sophus::SE3f _left_to_right_pose;
    float        _depth_threshold;
    float        _baseline_fx, _baseline;

    /*
     * Rectification stuff
     */
    cv::Mat _left_rectification_map_x, _left_rectification_map_y;
    cv::Mat _right_rectification_map_x, _right_rectification_map_y;

    /*
     * Inertial stuff
     */
    float        _gyroscope_noise, _accelerometer_noise;
    float        _gyroscope_random_walk, _accelerometer_random_walk;
    float        _imu_frequency;
    Sophus::SE3f _imu_to_camera_pose;
    bool         _insert_key_frames_when_lost;

    /*
     * RGBD stuff
     */
    float _depth_map_factor;

    /*
     * ORB stuff
     */
    int                      _feature_count;
    float                    _scale_factor;
    int                      _level_count;
    int                      _initial_fast_threshold, _minimum_fast_threshold;
    int                      _dbow_version{2};
    keypoint_detector_type   _keypoint_detector;
    keypoint_descriptor_type _keypoint_descriptor{keypoint_descriptor_type::orb};
    descriptor_matcher_type  _descriptor_matcher{descriptor_matcher_type::brute_force};
    bool                     _use_optical_flow{false};

    /*
     * Other stuff
     */
    float _far_point_threshold;
};
}; // namespace ORB_SLAM3

#endif // ORB_SLAM3_SETTINGS_H
