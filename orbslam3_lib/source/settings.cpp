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

#include "settings.h"

#include "CameraModels/KannalaBrandt8.h"
#include "CameraModels/pinhole.h"
#include "system.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>

#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

using namespace std;

namespace
{
struct kalibr_camera
{
    ORB_SLAM3::settings::camera_model_type type;
    std::vector<float>                     parameters; // ORB-SLAM3 camera model parameters
    std::vector<float>                     distortion; // OpenCV radtan coefficients, empty if none
    cv::Size                               resolution;
};

YAML::Node load_calibration(const std::filesystem::path &calibration_path)
{
    try
    {
        return YAML::LoadFile(calibration_path.string());
    }
    catch (const YAML::Exception &error)
    {
        throw std::runtime_error("Failed to read Kalibr calibration file " + calibration_path.string() + ": " + error.what());
    }
}

template <typename T> std::vector<T> read_values(const YAML::Node &node, const std::string &name, const std::size_t count, const std::filesystem::path &calibration_path)
{
    if (!node || !node.IsSequence() || node.size() != count)
    {
        throw std::runtime_error(name + " must contain " + std::to_string(count) + " values in calibration file: " + calibration_path.string());
    }

    std::vector<T> values;
    values.reserve(count);

    for (const YAML::Node &value : node)
    {
        values.push_back(value.as<T>());
    }

    return values;
}

Sophus::SE3f read_transform(const YAML::Node &transform, const std::string &name, const std::filesystem::path &calibration_path)
{
    if (!transform || !transform.IsSequence() || transform.size() != 4)
    {
        throw std::runtime_error(name + " must be a 4x4 matrix in calibration file: " + calibration_path.string());
    }

    Eigen::Matrix4d matrix;
    for (std::size_t row = 0; row < 4; ++row)
    {
        const std::vector<double> values = read_values<double>(transform[row], name, 4, calibration_path);
        for (std::size_t column = 0; column < 4; ++column)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) = values[column];
        }
    }

    if (!matrix.allFinite() || !matrix.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)))
    {
        throw std::runtime_error(name + " is not a valid homogeneous transform in calibration file: " + calibration_path.string());
    }

    return Sophus::SE3d(matrix).cast<float>();
}

kalibr_camera read_kalibr_camera(const YAML::Node &calibration, const std::string &name, const std::filesystem::path &calibration_path)
{
    const YAML::Node camera = calibration[name];

    if (!camera || !camera.IsMap())
    {
        throw std::runtime_error("Camera " + name + " not found in calibration file: " + calibration_path.string());
    }

    const std::string camera_model     = camera["camera_model"].as<std::string>("");
    const std::string distortion_model = camera["distortion_model"].as<std::string>("none");

    if (camera_model != "pinhole")
    {
        throw std::runtime_error(name + ".camera_model '" + camera_model + "' is not supported (expected pinhole) in calibration file: " + calibration_path.string());
    }

    kalibr_camera result;
    const auto    resolution = read_values<int>(camera["resolution"], name + ".resolution", 2, calibration_path);
    result.resolution        = cv::Size(resolution[0], resolution[1]);
    result.parameters        = read_values<float>(camera["intrinsics"], name + ".intrinsics", 4, calibration_path);

    if (distortion_model == "equidistant")
    {
        result.type                   = ORB_SLAM3::settings::kannala_brandt;
        const std::vector<float> k1k4 = read_values<float>(camera["distortion_coeffs"], name + ".distortion_coeffs", 4, calibration_path);
        result.parameters.insert(result.parameters.end(), k1k4.begin(), k1k4.end());
    }
    else if (distortion_model == "radtan")
    {
        result.type       = ORB_SLAM3::settings::pinhole;
        result.distortion = read_values<float>(camera["distortion_coeffs"], name + ".distortion_coeffs", 4, calibration_path);
    }
    else if (distortion_model == "none")
    {
        result.type = ORB_SLAM3::settings::pinhole;
    }
    else
    {
        throw std::runtime_error(name + ".distortion_model '" + distortion_model +
                                 "' is not supported (expected equidistant, radtan or none) in calibration file: " + calibration_path.string());
    }

    return result;
}

std::unique_ptr<ORB_SLAM3::geometric_camera> make_camera(const kalibr_camera &camera)
{
    if (camera.type == ORB_SLAM3::settings::kannala_brandt)
    {
        return std::make_unique<ORB_SLAM3::KannalaBrandt8>(camera.parameters);
    }

    return std::make_unique<ORB_SLAM3::pinhole>(camera.parameters);
}
} // namespace

namespace ORB_SLAM3
{

template <> float settings::read_parameter<float>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
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
            return 0.0f;
        }
    }
    else if (!node.isReal())
    {
        spdlog::error("{} parameter must be a real number, aborting...", name);
        exit(-1);
    }
    else
    {
        found = true;
        return node.real();
    }
}

template <> int settings::read_parameter<int>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
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
            return 0;
        }
    }
    else if (!node.isInt())
    {
        spdlog::error("{} parameter must be an integer number, aborting...", name);
        exit(-1);
    }
    else
    {
        found = true;
        return node.operator int();
    }
}

template <> string settings::read_parameter<string>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
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
            return string();
        }
    }
    else if (!node.isString())
    {
        spdlog::error("{} parameter must be a string, aborting...", name);
        exit(-1);
    }
    else
    {
        found = true;
        return node.string();
    }
}

template <> cv::Mat settings::read_parameter<cv::Mat>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
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
            return cv::Mat();
        }
    }
    else
    {
        found = true;
        return node.mat();
    }
}

settings::settings(const std::string &config_file, const int &sensor, const feature_overrides &overrides)
    : _need_to_undistort(false), _need_to_rectify(false), _need_to_resize_left(false), _need_to_resize_right(false)
{
    _sensor = sensor;

    // Open settings file
    cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
    if (!fSettings.isOpened())
    {
        spdlog::critical("Could not open configuration file at: {}. Aborting...", config_file);

        exit(-1);
    }
    else
    {
        spdlog::info("Loading settings from {}", config_file);
    }

    const cv::FileNode calibration = fSettings["data"]["calibration"];
    if (!calibration.isMap() || !calibration["file"].isString())
    {
        throw std::runtime_error("Settings entry data.calibration.file (Kalibr camchain YAML) is required in " + config_file);
    }
    _calibration_file = calibration["file"].string();
    if (_calibration_file.is_relative())
    {
        _calibration_file = std::filesystem::absolute(config_file).parent_path() / _calibration_file;
    }
    _calibration_file = _calibration_file.lexically_normal();

    read_cameras(fSettings);
    spdlog::info("Loaded camera calibration from {}", _calibration_file.string());

    // Read image info
    read_image_info(fSettings);
    spdlog::info("Loaded image info");

    if (_sensor == system::imu_monocular || _sensor == system::imu_stereo || _sensor == system::imu_rgbd)
    {
        read_imu(fSettings, config_file);
        spdlog::info("Loaded IMU calibration");
    }

    if (_sensor == system::rgbd || _sensor == system::imu_rgbd)
    {
        read_rgbd(fSettings);
        spdlog::info("Loaded RGB-D calibration");
    }

    read_features(fSettings, config_file, overrides);
    spdlog::info("Loaded feature settings");
    spdlog::info("Loaded Atlas settings");
    read_other_parameters(fSettings);
    spdlog::info("Loaded misc parameters");

    if (_need_to_rectify)
    {
        precompute_rectification_maps();
        spdlog::info("Computed rectification maps");
    }

    spdlog::info("----------------------------------");
}

void settings::read_cameras(cv::FileStorage &settings)
{
    const YAML::Node    calibration = load_calibration(_calibration_file);
    const kalibr_camera camera1     = read_kalibr_camera(calibration, "cam0", _calibration_file);

    _camera_type         = camera1.type;
    _original_image_size = camera1.resolution;
    _cameras.push_back(make_camera(camera1));
    _original_cameras.push_back(make_camera(camera1));
    _camera_1_distortion = camera1.distortion;

    if ((_sensor == system::monocular || _sensor == system::imu_monocular) && !_camera_1_distortion.empty())
    {
        _need_to_undistort = true;
    }

    if (_sensor != system::stereo && _sensor != system::imu_stereo)
    {
        return;
    }

    const kalibr_camera camera2 = read_kalibr_camera(calibration, "cam1", _calibration_file);

    if (camera2.type != _camera_type || camera2.resolution != _original_image_size)
    {
        throw std::runtime_error("cam0 and cam1 must share distortion model and resolution in calibration file: " + _calibration_file.string());
    }

    _cameras.push_back(make_camera(camera2));
    _original_cameras.push_back(make_camera(camera2));
    _camera_2_distortion = camera2.distortion;

    if (_camera_type == pinhole)
    {
        _need_to_rectify = true;
    }
    else
    {
        bool       found;
        const auto read_overlap = [&](const std::string &name, const int default_value)
        {
            const int value = read_parameter<int>(settings, name, found, false);
            return found ? value : default_value;
        };

        const int last_column                                     = _original_image_size.width - 1;
        static_cast<KannalaBrandt8 &>(*_cameras[0]).mvLappingArea = {read_overlap("Camera1.overlappingBegin", 0), read_overlap("Camera1.overlappingEnd", last_column)};
        static_cast<KannalaBrandt8 &>(*_cameras[1]).mvLappingArea = {read_overlap("Camera2.overlappingBegin", 0), read_overlap("Camera2.overlappingEnd", last_column)};
    }

    // Kalibr T_cn_cnm1 maps cam0 points into cam1, so left_to_right_pose is its inverse.
    _left_to_right_pose = read_transform(calibration["cam1"]["T_cn_cnm1"], "cam1.T_cn_cnm1", _calibration_file).inverse();
    _baseline           = _left_to_right_pose.translation().norm();
    _baseline_fx        = _baseline * _cameras[0]->get_parameter(0);

    bool found;
    _depth_threshold = read_parameter<float>(settings, "Stereo.ThDepth", found);
}

void settings::read_image_info(cv::FileStorage &fSettings)
{
    bool found;
    _new_image_size = _original_image_size;
    int newHeigh    = read_parameter<int>(fSettings, "Camera.newHeight", found, false);
    if (found)
    {
        _need_to_resize_left   = true;
        _new_image_size.height = newHeigh;

        if (!_need_to_rectify)
        {
            // Update calibration
            float scaleRowFactor = (float)_new_image_size.height / (float)_original_image_size.height;
            _cameras[0]->set_parameter(_cameras[0]->get_parameter(1) * scaleRowFactor, 1);
            _cameras[0]->set_parameter(_cameras[0]->get_parameter(3) * scaleRowFactor, 3);

            if (_sensor == system::stereo || _sensor == system::imu_stereo)
            {
                _cameras[1]->set_parameter(_cameras[1]->get_parameter(1) * scaleRowFactor, 1);
                _cameras[1]->set_parameter(_cameras[1]->get_parameter(3) * scaleRowFactor, 3);
            }
        }
    }

    int newWidth = read_parameter<int>(fSettings, "Camera.newWidth", found, false);
    if (found)
    {
        _need_to_resize_left  = true;
        _new_image_size.width = newWidth;

        if (!_need_to_rectify)
        {
            // Update calibration
            float scaleColFactor = (float)_new_image_size.width / (float)_original_image_size.width;
            _cameras[0]->set_parameter(_cameras[0]->get_parameter(0) * scaleColFactor, 0);
            _cameras[0]->set_parameter(_cameras[0]->get_parameter(2) * scaleColFactor, 2);

            if (_sensor == system::stereo || _sensor == system::imu_stereo)
            {
                _cameras[1]->set_parameter(_cameras[1]->get_parameter(0) * scaleColFactor, 0);
                _cameras[1]->set_parameter(_cameras[1]->get_parameter(2) * scaleColFactor, 2);

                if (_camera_type == kannala_brandt)
                {
                    static_cast<KannalaBrandt8 &>(*_cameras[0]).mvLappingArea[0] *= scaleColFactor;
                    static_cast<KannalaBrandt8 &>(*_cameras[0]).mvLappingArea[1] *= scaleColFactor;

                    static_cast<KannalaBrandt8 &>(*_cameras[1]).mvLappingArea[0] *= scaleColFactor;
                    static_cast<KannalaBrandt8 &>(*_cameras[1]).mvLappingArea[1] *= scaleColFactor;
                }
            }
        }
    }

    _fps = read_parameter<int>(fSettings, "Camera.fps", found);
    _rgb = (bool)read_parameter<int>(fSettings, "Camera.RGB", found);
}

void settings::read_imu(cv::FileStorage &settings, const std::string &config_file)
{
    for (const auto *key : {"IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk", "IMU.Frequency"})
    {
        if (!settings[key].empty())
        {
            throw std::runtime_error(std::string(key) + " is no longer supported in settings; use data.calibration.imu_file (Kalibr IMU YAML)");
        }
    }

    const auto imu_file = settings["data"]["calibration"]["imu_file"];
    if (!imu_file.isString())
    {
        throw std::runtime_error("Settings entry data.calibration.imu_file (Kalibr IMU YAML) is required in " + config_file);
    }

    auto imu_path = std::filesystem::path(imu_file.string());
    if (imu_path.is_relative())
    {
        imu_path = std::filesystem::absolute(config_file).parent_path() / imu_path;
    }

    imu_path               = imu_path.lexically_normal();
    const auto calibration = load_calibration(imu_path);
    const auto read_value  = [&](const std::string &key)
    {
        float value;
        try
        {
            value = calibration[key].as<float>();
        }
        catch (const YAML::Exception &error)
        {
            throw std::runtime_error("Invalid " + key + " in calibration file " + imu_path.string() + ": " + error.what());
        }
        if (!std::isfinite(value) || value <= 0.0f)
        {
            throw std::runtime_error(key + " must be finite and positive in calibration file: " + imu_path.string());
        }
        return value;
    };

    _gyroscope_noise           = read_value("gyroscope_noise_density");
    _accelerometer_noise       = read_value("accelerometer_noise_density");
    _gyroscope_random_walk     = read_value("gyroscope_random_walk");
    _accelerometer_random_walk = read_value("accelerometer_random_walk");
    _imu_frequency             = read_value("update_rate");
    spdlog::info("Loaded IMU calibration from {}", imu_path.string());

    // Kalibr T_cam_imu maps IMU points into cam0, so imu_to_camera_pose is its inverse.
    _imu_to_camera_pose = read_transform(load_calibration(_calibration_file)["cam0"]["T_cam_imu"], "cam0.T_cam_imu", _calibration_file).inverse();

    bool found;
    read_parameter<int>(settings, "IMU.InsertKFsWhenLost", found, false);
    if (found)
    {
        _insert_key_frames_when_lost = (bool)read_parameter<int>(settings, "IMU.InsertKFsWhenLost", found, false);
    }
    else
    {
        _insert_key_frames_when_lost = true;
    }
}

void settings::read_rgbd(cv::FileStorage &fSettings)
{
    bool found;

    _depth_map_factor = read_parameter<float>(fSettings, "RGBD.DepthMapFactor", found);
    _depth_threshold  = read_parameter<float>(fSettings, "Stereo.ThDepth", found);
    _baseline         = read_parameter<float>(fSettings, "Stereo.b", found);
    _baseline_fx      = _baseline * _cameras[0]->get_parameter(0);
}

void settings::read_features(cv::FileStorage &fSettings, const std::string &config_file, const feature_overrides &overrides)
{
    std::ifstream stream(config_file);
    if (stream.peek() == '%')
    {
        std::string directive;
        std::getline(stream, directive);
    }
    const auto root = YAML::Load(stream);
    _dbow_version   = overrides.dbow_version ? *overrides.dbow_version : root["dbow_version"] ? root["dbow_version"].as<int>() : 2;
    if (_dbow_version != 2 && _dbow_version != 3)
    {
        throw std::runtime_error("dbow_version must be 2 or 3");
    }
    const auto keypoint = root["keypoint"];
    if (!keypoint || !keypoint.IsMap())
    {
        throw std::runtime_error("keypoint must be a map containing detector and descriptor settings");
    }
    const auto detector = keypoint["detector"];
    if (!detector || !detector.IsMap())
    {
        throw std::runtime_error("keypoint.detector must be a map containing type, n_features, scale_factor, n_levels, initial_threshold, and minimum_threshold");
    }
    _feature_count          = detector["n_features"].as<int>();
    _scale_factor           = detector["scale_factor"].as<float>();
    _level_count            = detector["n_levels"].as<int>();
    _initial_fast_threshold = detector["initial_threshold"].as<int>();
    _minimum_fast_threshold = detector["minimum_threshold"].as<int>();
    if (_feature_count <= 0 || _level_count <= 0 || _level_count > 32 || !std::isfinite(_scale_factor) || _scale_factor <= 1 || _initial_fast_threshold < 0 || _minimum_fast_threshold < 0 ||
        _minimum_fast_threshold > _initial_fast_threshold)
    {
        throw std::runtime_error("Invalid keypoint.detector values: require positive features, 1..32 levels, finite scale_factor > 1, and 0 <= minimum_threshold <= initial_threshold");
    }
    const auto read_selection = [&]<typename ENUM_T_>(const std::string &section, const std::string &override_value, ENUM_T_ default_value)
    {
        const auto node = keypoint[section];
        if (node && !node.IsMap())
        {
            throw std::runtime_error("keypoint." + section + " must be a map");
        }
        const auto value  = !override_value.empty() ? override_value : node && node["type"] ? node["type"].as<std::string>() : std::string(magic_enum::enum_name(default_value));
        const auto parsed = magic_enum::enum_cast<ENUM_T_>(value, magic_enum::case_insensitive);
        if (!parsed)
        {
            throw std::runtime_error("Unknown keypoint." + section + ".type: " + value);
        }
        return *parsed;
    };
    _keypoint_detector           = read_selection("detector", overrides.detector, keypoint_detector_type::fast);
    _keypoint_descriptor         = read_selection("descriptor", overrides.descriptor, keypoint_descriptor_type::orb);
    const auto tracking_settings = root["tracking"];
    if (tracking_settings && !tracking_settings.IsMap())
    {
        throw std::runtime_error("tracking must be a map");
    }
    auto matcher_name = overrides.matcher.empty() ? std::string("brute_force") : overrides.matcher;
    if (overrides.matcher.empty() && tracking_settings && tracking_settings["matcher"])
    {
        const auto matcher_setting = tracking_settings["matcher"];
        if (!matcher_setting.IsScalar())
        {
            throw std::runtime_error("tracking.matcher must be a string");
        }
        matcher_name = matcher_setting.as<std::string>();
    }
    const auto matcher = magic_enum::enum_cast<descriptor_matcher_type>(matcher_name, magic_enum::case_insensitive);
    if (!matcher)
    {
        throw std::runtime_error("Unknown tracking.matcher: " + matcher_name);
    }
    _descriptor_matcher = *matcher;
    spdlog::info("Feature pipeline: detector={}, descriptor={}, matcher={}",
                 magic_enum::enum_name(_keypoint_detector),
                 magic_enum::enum_name(_keypoint_descriptor),
                 magic_enum::enum_name(_descriptor_matcher));

    const cv::FileNode tracking_node = fSettings["tracking"];
    if (!tracking_node.empty() && !tracking_node.isMap())
    {
        throw std::runtime_error("Settings entry tracking must be a map");
    }
    const cv::FileNode optical_flow_node = tracking_node["use_optical_flow"];
    int                use_optical_flow  = 0;

    if (!optical_flow_node.empty())
    {
        if (!optical_flow_node.isInt())
        {
            throw std::runtime_error("tracking.use_optical_flow must be 0 or 1");
        }
        use_optical_flow = optical_flow_node.operator int();
        if (use_optical_flow != 0 && use_optical_flow != 1)
        {
            throw std::runtime_error("tracking.use_optical_flow must be 0 or 1");
        }
    }

    _use_optical_flow = overrides.use_optical_flow.value_or(use_optical_flow == 1);
}

void settings::read_other_parameters(cv::FileStorage &fSettings)
{
    bool found;

    _far_point_threshold = read_parameter<float>(fSettings, "System.thFarPoints", found, false);
}

void settings::precompute_rectification_maps()
{
    // Precompute rectification maps, new calibrations, ...
    cv::Mat K1 = _cameras[0]->to_k();
    K1.convertTo(K1, CV_64F);
    cv::Mat K2 = _cameras[1]->to_k();
    K2.convertTo(K2, CV_64F);

    cv::Mat cvTlr;
    cv::eigen2cv(_left_to_right_pose.inverse().matrix3x4(), cvTlr);
    cv::Mat R12 = cvTlr.rowRange(0, 3).colRange(0, 3);
    R12.convertTo(R12, CV_64F);
    cv::Mat t12 = cvTlr.rowRange(0, 3).col(3);
    t12.convertTo(t12, CV_64F);

    cv::Mat R_r1_u1, R_r2_u2;
    cv::Mat P1, P2, Q;

    cv::stereoRectify(
        K1, camera_1_distortion_coefficients(), K2, camera_2_distortion_coefficients(), _new_image_size, R12, t12, R_r1_u1, R_r2_u2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, _new_image_size);
    cv::initUndistortRectifyMap(
        K1, camera_1_distortion_coefficients(), R_r1_u1, P1.rowRange(0, 3).colRange(0, 3), _new_image_size, CV_32F, _left_rectification_map_x, _left_rectification_map_y);
    cv::initUndistortRectifyMap(
        K2, camera_2_distortion_coefficients(), R_r2_u2, P2.rowRange(0, 3).colRange(0, 3), _new_image_size, CV_32F, _right_rectification_map_x, _right_rectification_map_y);

    // Update calibration
    _cameras[0]->set_parameter(P1.at<double>(0, 0), 0);
    _cameras[0]->set_parameter(P1.at<double>(1, 1), 1);
    _cameras[0]->set_parameter(P1.at<double>(0, 2), 2);
    _cameras[0]->set_parameter(P1.at<double>(1, 2), 3);
    _cameras[1]->set_parameter(P2.at<double>(0, 0), 0);
    _cameras[1]->set_parameter(P2.at<double>(1, 1), 1);
    _cameras[1]->set_parameter(P2.at<double>(0, 2), 2);
    _cameras[1]->set_parameter(P2.at<double>(1, 2), 3);

    // Update baseline_fx
    _baseline_fx = _baseline * P1.at<double>(0, 0);

    // Update relative pose between camera 1 and IMU if necessary
    if (_sensor == system::imu_stereo)
    {
        Eigen::Matrix3f eigenR_r1_u1;
        cv::cv2eigen(R_r1_u1, eigenR_r1_u1);
        Sophus::SE3f T_r1_u1(eigenR_r1_u1, Eigen::Vector3f::Zero());
        _imu_to_camera_pose = _imu_to_camera_pose * T_r1_u1.inverse();
    }
}

ostream &operator<<(std::ostream &output, const settings &settings)
{
    output << "SLAM settings: " << endl;
    output << "\t-Calibration file: " << settings._calibration_file.string() << endl;

    output << "\t-Camera 1 parameters (";
    if (settings._camera_type == settings::pinhole)
    {
        output << "Pinhole";
    }
    else
    {
        output << "Kannala-Brandt";
    }
    output << ")" << ": [";
    for (size_t i = 0; i < settings._original_cameras[0]->size(); i++)
    {
        output << " " << settings._original_cameras[0]->get_parameter(i);
    }
    output << " ]" << endl;

    if (!settings._camera_1_distortion.empty())
    {
        output << "\t-Camera 1 distortion parameters: [ ";
        for (float d : settings._camera_1_distortion)
        {
            output << " " << d;
        }
        output << " ]" << endl;
    }

    if (settings._sensor == system::stereo || settings._sensor == system::imu_stereo)
    {
        output << "\t-Camera 2 parameters (";
        if (settings._camera_type == settings::pinhole)
        {
            output << "Pinhole";
        }
        else
        {
            output << "Kannala-Brandt";
        }
        output << "" << ": [";
        for (size_t i = 0; i < settings._original_cameras[1]->size(); i++)
        {
            output << " " << settings._original_cameras[1]->get_parameter(i);
        }
        output << " ]" << endl;

        if (!settings._camera_2_distortion.empty())
        {
            output << "\t-Camera 1 distortion parameters: [ ";
            for (float d : settings._camera_2_distortion)
            {
                output << " " << d;
            }
            output << " ]" << endl;
        }
    }

    output << "\t-Original image size: [ " << settings._original_image_size.width << " , " << settings._original_image_size.height << " ]" << endl;
    output << "\t-Current image size: [ " << settings._new_image_size.width << " , " << settings._new_image_size.height << " ]" << endl;

    if (settings._need_to_rectify)
    {
        output << "\t-Camera 1 parameters after rectification: [ ";
        for (size_t i = 0; i < settings._cameras[0]->size(); i++)
        {
            output << " " << settings._cameras[0]->get_parameter(i);
        }
        output << " ]" << endl;
    }
    else if (settings._need_to_resize_left)
    {
        output << "\t-Camera 1 parameters after resize: [ ";
        for (size_t i = 0; i < settings._cameras[0]->size(); i++)
        {
            output << " " << settings._cameras[0]->get_parameter(i);
        }
        output << " ]" << endl;

        if ((settings._sensor == system::stereo || settings._sensor == system::imu_stereo) && settings._camera_type == settings::kannala_brandt)
        {
            output << "\t-Camera 2 parameters after resize: [ ";
            for (size_t i = 0; i < settings._cameras[1]->size(); i++)
            {
                output << " " << settings._cameras[1]->get_parameter(i);
            }
            output << " ]" << endl;
        }
    }

    output << "\t-Sequence FPS: " << settings._fps << endl;

    // Stereo stuff
    if (settings._sensor == system::stereo || settings._sensor == system::imu_stereo)
    {
        output << "\t-Stereo baseline: " << settings._baseline << endl;
        output << "\t-Stereo depth threshold : " << settings._depth_threshold << endl;

        if (settings._camera_type == settings::kannala_brandt)
        {
            auto vOverlapping1 = static_cast<KannalaBrandt8 &>(*settings._cameras[0]).mvLappingArea;
            auto vOverlapping2 = static_cast<KannalaBrandt8 &>(*settings._cameras[1]).mvLappingArea;
            output << "\t-Camera 1 overlapping area: [ " << vOverlapping1[0] << " , " << vOverlapping1[1] << " ]" << endl;
            output << "\t-Camera 2 overlapping area: [ " << vOverlapping2[0] << " , " << vOverlapping2[1] << " ]" << endl;
        }
    }

    if (settings._sensor == system::imu_monocular || settings._sensor == system::imu_stereo || settings._sensor == system::imu_rgbd)
    {
        output << "\t-Gyro noise: " << settings._gyroscope_noise << endl;
        output << "\t-Accelerometer noise: " << settings._accelerometer_noise << endl;
        output << "\t-Gyro walk: " << settings._gyroscope_random_walk << endl;
        output << "\t-Accelerometer walk: " << settings._accelerometer_random_walk << endl;
        output << "\t-IMU frequency: " << settings._imu_frequency << endl;
    }

    if (settings._sensor == system::rgbd || settings._sensor == system::imu_rgbd)
    {
        output << "\t-RGB-D depth map factor: " << settings._depth_map_factor << endl;
    }

    output << "\t-Features per image: " << settings._feature_count << endl;
    output << "\t-Feature scale factor: " << settings._scale_factor << endl;
    output << "\t-Feature number of scales: " << settings._level_count << endl;
    output << "\t-Initial FAST threshold: " << settings._initial_fast_threshold << endl;
    output << "\t-Min FAST threshold: " << settings._minimum_fast_threshold << endl;
    output << "\t-Keypoint detector: " << magic_enum::enum_name(settings._keypoint_detector) << endl;

    return output;
}
}; // namespace ORB_SLAM3
