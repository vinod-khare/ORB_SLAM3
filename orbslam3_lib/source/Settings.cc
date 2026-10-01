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

#include "Settings.h"

#include "CameraModels/KannalaBrandt8.h"
#include "CameraModels/Pinhole.h"
#include "System.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>

#include <magic_enum/magic_enum.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;

namespace
{
struct kalibr_camera
{
    ORB_SLAM3::Settings::CameraType type;
    std::vector<float>              parameters; // ORB-SLAM3 camera model parameters
    std::vector<float>              distortion; // OpenCV radtan coefficients, empty if none
    cv::Size                        resolution;
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
        result.type                   = ORB_SLAM3::Settings::KannalaBrandt;
        const std::vector<float> k1k4 = read_values<float>(camera["distortion_coeffs"], name + ".distortion_coeffs", 4, calibration_path);
        result.parameters.insert(result.parameters.end(), k1k4.begin(), k1k4.end());
    }
    else if (distortion_model == "radtan")
    {
        result.type       = ORB_SLAM3::Settings::PinHole;
        result.distortion = read_values<float>(camera["distortion_coeffs"], name + ".distortion_coeffs", 4, calibration_path);
    }
    else if (distortion_model == "none")
    {
        result.type = ORB_SLAM3::Settings::PinHole;
    }
    else
    {
        throw std::runtime_error(name + ".distortion_model '" + distortion_model +
                                 "' is not supported (expected equidistant, radtan or none) in calibration file: " + calibration_path.string());
    }

    return result;
}

ORB_SLAM3::GeometricCamera *make_camera(const kalibr_camera &camera)
{
    if (camera.type == ORB_SLAM3::Settings::KannalaBrandt)
    {
        return new ORB_SLAM3::KannalaBrandt8(camera.parameters);
    }

    return new ORB_SLAM3::Pinhole(camera.parameters);
}
} // namespace

namespace ORB_SLAM3
{

template <> float Settings::readParameter<float>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
{
    cv::FileNode node = fSettings[name];
    if (node.empty())
    {
        if (required)
        {
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else
        {
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0.0f;
        }
    }
    else if (!node.isReal())
    {
        std::cerr << name << " parameter must be a real number, aborting..." << std::endl;
        exit(-1);
    }
    else
    {
        found = true;
        return node.real();
    }
}

template <> int Settings::readParameter<int>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
{
    cv::FileNode node = fSettings[name];
    if (node.empty())
    {
        if (required)
        {
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else
        {
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return 0;
        }
    }
    else if (!node.isInt())
    {
        std::cerr << name << " parameter must be an integer number, aborting..." << std::endl;
        exit(-1);
    }
    else
    {
        found = true;
        return node.operator int();
    }
}

template <> string Settings::readParameter<string>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
{
    cv::FileNode node = fSettings[name];
    if (node.empty())
    {
        if (required)
        {
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else
        {
            std::cerr << name << " optional parameter does not exist..." << std::endl;
            found = false;
            return string();
        }
    }
    else if (!node.isString())
    {
        std::cerr << name << " parameter must be a string, aborting..." << std::endl;
        exit(-1);
    }
    else
    {
        found = true;
        return node.string();
    }
}

template <> cv::Mat Settings::readParameter<cv::Mat>(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required)
{
    cv::FileNode node = fSettings[name];
    if (node.empty())
    {
        if (required)
        {
            std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
            exit(-1);
        }
        else
        {
            std::cerr << name << " optional parameter does not exist..." << std::endl;
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

Settings::Settings(const std::string &config_file, const int &sensor) : bNeedToUndistort_(false), bNeedToRectify_(false), bNeedToResize1_(false), bNeedToResize2_(false)
{
    sensor_ = sensor;

    // Open settings file
    cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
    if (!fSettings.isOpened())
    {
        cerr << "[ERROR]: could not open configuration file at: " << config_file << endl;
        cerr << "Aborting..." << endl;

        exit(-1);
    }
    else
    {
        cout << "Loading settings from " << config_file << endl;
    }

    const cv::FileNode calibration = fSettings["calibration"];
    if (!calibration.isMap() || !calibration["file"].isString())
    {
        throw std::runtime_error("Settings entry calibration.file (Kalibr camchain YAML) is required in " + config_file);
    }
    _calibration_file = calibration["file"].string();
    if (_calibration_file.is_relative())
    {
        _calibration_file = std::filesystem::absolute(config_file).parent_path() / _calibration_file;
    }
    _calibration_file = _calibration_file.lexically_normal();

    read_cameras(fSettings);
    cout << "\t-Loaded camera calibration from " << _calibration_file.string() << endl;

    // Read image info
    readImageInfo(fSettings);
    cout << "\t-Loaded image info" << endl;

    if (sensor_ == System::IMU_MONOCULAR || sensor_ == System::IMU_STEREO || sensor_ == System::IMU_RGBD)
    {
        read_imu(fSettings);
        cout << "\t-Loaded IMU calibration" << endl;
    }

    if (sensor_ == System::RGBD || sensor_ == System::IMU_RGBD)
    {
        readRGBD(fSettings);
        cout << "\t-Loaded RGB-D calibration" << endl;
    }

    readORB(fSettings);
    cout << "\t-Loaded ORB settings" << endl;
    readViewer(fSettings);
    cout << "\t-Loaded viewer settings" << endl;
    readLoadAndSave(fSettings);
    cout << "\t-Loaded Atlas settings" << endl;
    readOtherParameters(fSettings);
    cout << "\t-Loaded misc parameters" << endl;

    if (bNeedToRectify_)
    {
        precomputeRectificationMaps();
        cout << "\t-Computed rectification maps" << endl;
    }

    cout << "----------------------------------" << endl;
}

void Settings::read_cameras(cv::FileStorage &settings)
{
    const YAML::Node    calibration = load_calibration(_calibration_file);
    const kalibr_camera camera1     = read_kalibr_camera(calibration, "cam0", _calibration_file);

    cameraType_                     = camera1.type;
    originalImSize_                 = camera1.resolution;
    calibration1_                   = make_camera(camera1);
    originalCalib1_.reset(make_camera(camera1));
    vPinHoleDistorsion1_ = camera1.distortion;

    if ((sensor_ == System::MONOCULAR || sensor_ == System::IMU_MONOCULAR) && !vPinHoleDistorsion1_.empty())
    {
        bNeedToUndistort_ = true;
    }

    if (sensor_ != System::STEREO && sensor_ != System::IMU_STEREO)
    {
        return;
    }

    const kalibr_camera camera2 = read_kalibr_camera(calibration, "cam1", _calibration_file);

    if (camera2.type != cameraType_ || camera2.resolution != originalImSize_)
    {
        throw std::runtime_error("cam0 and cam1 must share distortion model and resolution in calibration file: " + _calibration_file.string());
    }

    calibration2_ = make_camera(camera2);
    originalCalib2_.reset(make_camera(camera2));
    vPinHoleDistorsion2_ = camera2.distortion;

    if (cameraType_ == PinHole)
    {
        bNeedToRectify_ = true;
    }
    else
    {
        bool       found;
        const auto read_overlap = [&](const std::string &name, const int default_value)
        {
            const int value = readParameter<int>(settings, name, found, false);
            return found ? value : default_value;
        };

        const int last_column                                       = originalImSize_.width - 1;
        static_cast<KannalaBrandt8 *>(calibration1_)->mvLappingArea = {read_overlap("Camera1.overlappingBegin", 0), read_overlap("Camera1.overlappingEnd", last_column)};
        static_cast<KannalaBrandt8 *>(calibration2_)->mvLappingArea = {read_overlap("Camera2.overlappingBegin", 0), read_overlap("Camera2.overlappingEnd", last_column)};
    }

    // Kalibr T_cn_cnm1 maps cam0 points into cam1, so Tlr is its inverse.
    Tlr_ = read_transform(calibration["cam1"]["T_cn_cnm1"], "cam1.T_cn_cnm1", _calibration_file).inverse();
    b_   = Tlr_.translation().norm();
    bf_  = b_ * calibration1_->getParameter(0);

    bool found;
    thDepth_ = readParameter<float>(settings, "Stereo.ThDepth", found);
}

void Settings::readImageInfo(cv::FileStorage &fSettings)
{
    bool found;
    newImSize_   = originalImSize_;
    int newHeigh = readParameter<int>(fSettings, "Camera.newHeight", found, false);
    if (found)
    {
        bNeedToResize1_   = true;
        newImSize_.height = newHeigh;

        if (!bNeedToRectify_)
        {
            // Update calibration
            float scaleRowFactor = (float)newImSize_.height / (float)originalImSize_.height;
            calibration1_->setParameter(calibration1_->getParameter(1) * scaleRowFactor, 1);
            calibration1_->setParameter(calibration1_->getParameter(3) * scaleRowFactor, 3);

            if (sensor_ == System::STEREO || sensor_ == System::IMU_STEREO)
            {
                calibration2_->setParameter(calibration2_->getParameter(1) * scaleRowFactor, 1);
                calibration2_->setParameter(calibration2_->getParameter(3) * scaleRowFactor, 3);
            }
        }
    }

    int newWidth = readParameter<int>(fSettings, "Camera.newWidth", found, false);
    if (found)
    {
        bNeedToResize1_  = true;
        newImSize_.width = newWidth;

        if (!bNeedToRectify_)
        {
            // Update calibration
            float scaleColFactor = (float)newImSize_.width / (float)originalImSize_.width;
            calibration1_->setParameter(calibration1_->getParameter(0) * scaleColFactor, 0);
            calibration1_->setParameter(calibration1_->getParameter(2) * scaleColFactor, 2);

            if (sensor_ == System::STEREO || sensor_ == System::IMU_STEREO)
            {
                calibration2_->setParameter(calibration2_->getParameter(0) * scaleColFactor, 0);
                calibration2_->setParameter(calibration2_->getParameter(2) * scaleColFactor, 2);

                if (cameraType_ == KannalaBrandt)
                {
                    static_cast<KannalaBrandt8 *>(calibration1_)->mvLappingArea[0] *= scaleColFactor;
                    static_cast<KannalaBrandt8 *>(calibration1_)->mvLappingArea[1] *= scaleColFactor;

                    static_cast<KannalaBrandt8 *>(calibration2_)->mvLappingArea[0] *= scaleColFactor;
                    static_cast<KannalaBrandt8 *>(calibration2_)->mvLappingArea[1] *= scaleColFactor;
                }
            }
        }
    }

    fps_  = readParameter<int>(fSettings, "Camera.fps", found);
    bRGB_ = (bool)readParameter<int>(fSettings, "Camera.RGB", found);
}

void Settings::read_imu(cv::FileStorage &settings)
{
    bool found;
    noiseGyro_    = readParameter<float>(settings, "IMU.NoiseGyro", found);
    noiseAcc_     = readParameter<float>(settings, "IMU.NoiseAcc", found);
    gyroWalk_     = readParameter<float>(settings, "IMU.GyroWalk", found);
    accWalk_      = readParameter<float>(settings, "IMU.AccWalk", found);
    imuFrequency_ = readParameter<float>(settings, "IMU.Frequency", found);

    // Kalibr T_cam_imu maps IMU points into cam0, so Tbc is its inverse.
    Tbc_          = read_transform(load_calibration(_calibration_file)["cam0"]["T_cam_imu"], "cam0.T_cam_imu", _calibration_file).inverse();

    readParameter<int>(settings, "IMU.InsertKFsWhenLost", found, false);
    if (found)
    {
        insertKFsWhenLost_ = (bool)readParameter<int>(settings, "IMU.InsertKFsWhenLost", found, false);
    }
    else
    {
        insertKFsWhenLost_ = true;
    }
}

void Settings::readRGBD(cv::FileStorage &fSettings)
{
    bool found;

    depthMapFactor_ = readParameter<float>(fSettings, "RGBD.DepthMapFactor", found);
    thDepth_        = readParameter<float>(fSettings, "Stereo.ThDepth", found);
    b_              = readParameter<float>(fSettings, "Stereo.b", found);
    bf_             = b_ * calibration1_->getParameter(0);
}

void Settings::readORB(cv::FileStorage &fSettings)
{
    bool found;

    nFeatures_            = readParameter<int>(fSettings, "ORBextractor.nFeatures", found);
    scaleFactor_          = readParameter<float>(fSettings, "ORBextractor.scaleFactor", found);
    nLevels_              = readParameter<int>(fSettings, "ORBextractor.nLevels", found);
    initThFAST_           = readParameter<int>(fSettings, "ORBextractor.iniThFAST", found);
    minThFAST_            = readParameter<int>(fSettings, "ORBextractor.minThFAST", found);

    const string detector = readParameter<string>(fSettings, "ORBextractor.detector", found, false);
    _keypoint_detector    = keypoint_detector_type::fast;
    if (found)
    {
        const auto parsed = magic_enum::enum_cast<keypoint_detector_type>(detector, magic_enum::case_insensitive);
        if (!parsed)
        {
            std::cerr << "ORBextractor.detector: unknown keypoint detector '" << detector << "', expected one of:";
            for (const auto name : magic_enum::enum_names<keypoint_detector_type>())
            {
                std::cerr << " " << name;
            }
            std::cerr << ", aborting..." << std::endl;
            exit(-1);
        }
        _keypoint_detector = *parsed;
    }

    const cv::FileNode optical_flow_node = fSettings["Tracking.useOpticalFlow"];
    int                use_optical_flow  = 0;

    if (!optical_flow_node.empty())
    {
        if (!optical_flow_node.isInt())
        {
            std::cerr << "Tracking.useOpticalFlow must be 0 or 1, aborting..." << std::endl;
            exit(-1);
        }

        use_optical_flow = optical_flow_node.operator int();

        if (use_optical_flow != 0 && use_optical_flow != 1)
        {
            std::cerr << "Tracking.useOpticalFlow must be 0 or 1, aborting..." << std::endl;
            exit(-1);
        }
    }

    _use_optical_flow = use_optical_flow == 1;
}

void Settings::readViewer(cv::FileStorage &fSettings)
{
    bool found;

    keyFrameSize_      = readParameter<float>(fSettings, "Viewer.KeyFrameSize", found);
    keyFrameLineWidth_ = readParameter<float>(fSettings, "Viewer.KeyFrameLineWidth", found);
    graphLineWidth_    = readParameter<float>(fSettings, "Viewer.GraphLineWidth", found);
    pointSize_         = readParameter<float>(fSettings, "Viewer.PointSize", found);
    cameraSize_        = readParameter<float>(fSettings, "Viewer.CameraSize", found);
    cameraLineWidth_   = readParameter<float>(fSettings, "Viewer.CameraLineWidth", found);
    viewPointX_        = readParameter<float>(fSettings, "Viewer.ViewpointX", found);
    viewPointY_        = readParameter<float>(fSettings, "Viewer.ViewpointY", found);
    viewPointZ_        = readParameter<float>(fSettings, "Viewer.ViewpointZ", found);
    viewPointF_        = readParameter<float>(fSettings, "Viewer.ViewpointF", found);
    imageViewerScale_  = readParameter<float>(fSettings, "Viewer.imageViewScale", found, false);

    if (!found)
    {
        imageViewerScale_ = 1.0f;
    }
}

void Settings::readLoadAndSave(cv::FileStorage &fSettings)
{
    bool found;

    sLoadFrom_ = readParameter<string>(fSettings, "System.LoadAtlasFromFile", found, false);
    sSaveto_   = readParameter<string>(fSettings, "System.SaveAtlasToFile", found, false);
}

void Settings::readOtherParameters(cv::FileStorage &fSettings)
{
    bool found;

    thFarPoints_ = readParameter<float>(fSettings, "System.thFarPoints", found, false);
}

void Settings::precomputeRectificationMaps()
{
    // Precompute rectification maps, new calibrations, ...
    cv::Mat K1 = static_cast<Pinhole *>(calibration1_)->toK();
    K1.convertTo(K1, CV_64F);
    cv::Mat K2 = static_cast<Pinhole *>(calibration2_)->toK();
    K2.convertTo(K2, CV_64F);

    cv::Mat cvTlr;
    cv::eigen2cv(Tlr_.inverse().matrix3x4(), cvTlr);
    cv::Mat R12 = cvTlr.rowRange(0, 3).colRange(0, 3);
    R12.convertTo(R12, CV_64F);
    cv::Mat t12 = cvTlr.rowRange(0, 3).col(3);
    t12.convertTo(t12, CV_64F);

    cv::Mat R_r1_u1, R_r2_u2;
    cv::Mat P1, P2, Q;

    cv::stereoRectify(K1, camera1DistortionCoef(), K2, camera2DistortionCoef(), newImSize_, R12, t12, R_r1_u1, R_r2_u2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, newImSize_);
    cv::initUndistortRectifyMap(K1, camera1DistortionCoef(), R_r1_u1, P1.rowRange(0, 3).colRange(0, 3), newImSize_, CV_32F, M1l_, M2l_);
    cv::initUndistortRectifyMap(K2, camera2DistortionCoef(), R_r2_u2, P2.rowRange(0, 3).colRange(0, 3), newImSize_, CV_32F, M1r_, M2r_);

    // Update calibration
    calibration1_->setParameter(P1.at<double>(0, 0), 0);
    calibration1_->setParameter(P1.at<double>(1, 1), 1);
    calibration1_->setParameter(P1.at<double>(0, 2), 2);
    calibration1_->setParameter(P1.at<double>(1, 2), 3);

    // Update bf
    bf_ = b_ * P1.at<double>(0, 0);

    // Update relative pose between camera 1 and IMU if necessary
    if (sensor_ == System::IMU_STEREO)
    {
        Eigen::Matrix3f eigenR_r1_u1;
        cv::cv2eigen(R_r1_u1, eigenR_r1_u1);
        Sophus::SE3f T_r1_u1(eigenR_r1_u1, Eigen::Vector3f::Zero());
        Tbc_ = Tbc_ * T_r1_u1.inverse();
    }
}

ostream &operator<<(std::ostream &output, const Settings &settings)
{
    output << "SLAM settings: " << endl;
    output << "\t-Calibration file: " << settings._calibration_file.string() << endl;

    output << "\t-Camera 1 parameters (";
    if (settings.cameraType_ == Settings::PinHole)
    {
        output << "Pinhole";
    }
    else
    {
        output << "Kannala-Brandt";
    }
    output << ")" << ": [";
    for (size_t i = 0; i < settings.originalCalib1_->size(); i++)
    {
        output << " " << settings.originalCalib1_->getParameter(i);
    }
    output << " ]" << endl;

    if (!settings.vPinHoleDistorsion1_.empty())
    {
        output << "\t-Camera 1 distortion parameters: [ ";
        for (float d : settings.vPinHoleDistorsion1_)
        {
            output << " " << d;
        }
        output << " ]" << endl;
    }

    if (settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO)
    {
        output << "\t-Camera 2 parameters (";
        if (settings.cameraType_ == Settings::PinHole)
        {
            output << "Pinhole";
        }
        else
        {
            output << "Kannala-Brandt";
        }
        output << "" << ": [";
        for (size_t i = 0; i < settings.originalCalib2_->size(); i++)
        {
            output << " " << settings.originalCalib2_->getParameter(i);
        }
        output << " ]" << endl;

        if (!settings.vPinHoleDistorsion2_.empty())
        {
            output << "\t-Camera 1 distortion parameters: [ ";
            for (float d : settings.vPinHoleDistorsion2_)
            {
                output << " " << d;
            }
            output << " ]" << endl;
        }
    }

    output << "\t-Original image size: [ " << settings.originalImSize_.width << " , " << settings.originalImSize_.height << " ]" << endl;
    output << "\t-Current image size: [ " << settings.newImSize_.width << " , " << settings.newImSize_.height << " ]" << endl;

    if (settings.bNeedToRectify_)
    {
        output << "\t-Camera 1 parameters after rectification: [ ";
        for (size_t i = 0; i < settings.calibration1_->size(); i++)
        {
            output << " " << settings.calibration1_->getParameter(i);
        }
        output << " ]" << endl;
    }
    else if (settings.bNeedToResize1_)
    {
        output << "\t-Camera 1 parameters after resize: [ ";
        for (size_t i = 0; i < settings.calibration1_->size(); i++)
        {
            output << " " << settings.calibration1_->getParameter(i);
        }
        output << " ]" << endl;

        if ((settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO) && settings.cameraType_ == Settings::KannalaBrandt)
        {
            output << "\t-Camera 2 parameters after resize: [ ";
            for (size_t i = 0; i < settings.calibration2_->size(); i++)
            {
                output << " " << settings.calibration2_->getParameter(i);
            }
            output << " ]" << endl;
        }
    }

    output << "\t-Sequence FPS: " << settings.fps_ << endl;

    // Stereo stuff
    if (settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO)
    {
        output << "\t-Stereo baseline: " << settings.b_ << endl;
        output << "\t-Stereo depth threshold : " << settings.thDepth_ << endl;

        if (settings.cameraType_ == Settings::KannalaBrandt)
        {
            auto vOverlapping1 = static_cast<KannalaBrandt8 *>(settings.calibration1_)->mvLappingArea;
            auto vOverlapping2 = static_cast<KannalaBrandt8 *>(settings.calibration2_)->mvLappingArea;
            output << "\t-Camera 1 overlapping area: [ " << vOverlapping1[0] << " , " << vOverlapping1[1] << " ]" << endl;
            output << "\t-Camera 2 overlapping area: [ " << vOverlapping2[0] << " , " << vOverlapping2[1] << " ]" << endl;
        }
    }

    if (settings.sensor_ == System::IMU_MONOCULAR || settings.sensor_ == System::IMU_STEREO || settings.sensor_ == System::IMU_RGBD)
    {
        output << "\t-Gyro noise: " << settings.noiseGyro_ << endl;
        output << "\t-Accelerometer noise: " << settings.noiseAcc_ << endl;
        output << "\t-Gyro walk: " << settings.gyroWalk_ << endl;
        output << "\t-Accelerometer walk: " << settings.accWalk_ << endl;
        output << "\t-IMU frequency: " << settings.imuFrequency_ << endl;
    }

    if (settings.sensor_ == System::RGBD || settings.sensor_ == System::IMU_RGBD)
    {
        output << "\t-RGB-D depth map factor: " << settings.depthMapFactor_ << endl;
    }

    output << "\t-Features per image: " << settings.nFeatures_ << endl;
    output << "\t-ORB scale factor: " << settings.scaleFactor_ << endl;
    output << "\t-ORB number of scales: " << settings.nLevels_ << endl;
    output << "\t-Initial FAST threshold: " << settings.initThFAST_ << endl;
    output << "\t-Min FAST threshold: " << settings.minThFAST_ << endl;
    output << "\t-Keypoint detector: " << magic_enum::enum_name(settings._keypoint_detector) << endl;

    return output;
}
}; // namespace ORB_SLAM3
