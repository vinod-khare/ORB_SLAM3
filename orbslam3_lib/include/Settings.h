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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <filesystem>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

class System;

class Settings
{
  public:
    /*
     * Enum for the different camera types implemented
     */
    enum CameraType
    {
        PinHole       = 0,
        KannalaBrandt = 2
    };

    /*
     * Delete default constructor
     */
    Settings() = delete;

    /*
     * Constructor from file
     */
    Settings(const std::string &config_file, const int &sensor, const feature_overrides &overrides = {});

    /*
     * Ostream operator overloading to dump settings to the terminal
     */
    friend std::ostream &operator<<(std::ostream &output, const Settings &s);

    /*
     * Getter methods
     */
    CameraType       cameraType() { return cameraType_; }
    GeometricCamera *camera1() { return calibration1_; }
    GeometricCamera *camera2() { return calibration2_; }
    cv::Mat          camera1DistortionCoef() { return cv::Mat(vPinHoleDistorsion1_.size(), 1, CV_32F, vPinHoleDistorsion1_.data()); }
    cv::Mat          camera2DistortionCoef() { return cv::Mat(vPinHoleDistorsion2_.size(), 1, CV_32F, vPinHoleDistorsion2_.data()); }

    Sophus::SE3f Tlr() { return Tlr_; }
    float        bf() { return bf_; }
    float        b() { return b_; }
    float        thDepth() { return thDepth_; }

    bool needToUndistort() { return bNeedToUndistort_; }

    cv::Size newImSize() { return newImSize_; }
    float    fps() { return fps_; }
    bool     rgb() { return bRGB_; }
    bool     needToResize() { return bNeedToResize1_; }
    bool     needToRectify() { return bNeedToRectify_; }

    float        noiseGyro() { return noiseGyro_; }
    float        noiseAcc() { return noiseAcc_; }
    float        gyroWalk() { return gyroWalk_; }
    float        accWalk() { return accWalk_; }
    float        imuFrequency() { return imuFrequency_; }
    Sophus::SE3f Tbc() { return Tbc_; }
    bool         insertKFsWhenLost() { return insertKFsWhenLost_; }

    float depthMapFactor() { return depthMapFactor_; }

    int                      nFeatures() { return nFeatures_; }
    int                      nLevels() { return nLevels_; }
    float                    initThFAST() { return initThFAST_; }
    float                    minThFAST() { return minThFAST_; }
    float                    scaleFactor() { return scaleFactor_; }
    keypoint_detector_type   keypoint_detector() const { return _keypoint_detector; }
    keypoint_descriptor_type keypoint_descriptor() const { return _keypoint_descriptor; }
    descriptor_matcher_type  matcher_backend() const { return _descriptor_matcher; }
    bool                     use_optical_flow() const { return _use_optical_flow; }
    int                      dbow_version() const { return _dbow_version; }

    float thFarPoints() { return thFarPoints_; }

    cv::Mat M1l() { return M1l_; }
    cv::Mat M2l() { return M2l_; }
    cv::Mat M1r() { return M1r_; }
    cv::Mat M2r() { return M2r_; }

  private:
    template <typename T> T readParameter(cv::FileStorage &fSettings, const std::string &name, bool &found, const bool required = true)
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
    void readImageInfo(cv::FileStorage &fSettings);
    void read_imu(cv::FileStorage &settings, const std::string &config_file);
    void readRGBD(cv::FileStorage &fSettings);
    void read_features(cv::FileStorage &fSettings, const std::string &config_file, const feature_overrides &overrides);
    void readOtherParameters(cv::FileStorage &fSettings);

    void precomputeRectificationMaps();

    int                   sensor_;
    std::filesystem::path _calibration_file; // Kalibr camchain file
    CameraType            cameraType_;       // Camera type

    /*
     * Visual stuff
     */
    GeometricCamera                 *calibration1_{}, *calibration2_{}; // Camera calibration
    std::unique_ptr<GeometricCamera> originalCalib1_, originalCalib2_;
    std::vector<float>               vPinHoleDistorsion1_, vPinHoleDistorsion2_;

    cv::Size originalImSize_, newImSize_;
    float    fps_;
    bool     bRGB_;

    bool bNeedToUndistort_;
    bool bNeedToRectify_;
    bool bNeedToResize1_, bNeedToResize2_;

    Sophus::SE3f Tlr_;
    float        thDepth_;
    float        bf_, b_;

    /*
     * Rectification stuff
     */
    cv::Mat M1l_, M2l_;
    cv::Mat M1r_, M2r_;

    /*
     * Inertial stuff
     */
    float        noiseGyro_, noiseAcc_;
    float        gyroWalk_, accWalk_;
    float        imuFrequency_;
    Sophus::SE3f Tbc_;
    bool         insertKFsWhenLost_;

    /*
     * RGBD stuff
     */
    float depthMapFactor_;

    /*
     * ORB stuff
     */
    int                      nFeatures_;
    float                    scaleFactor_;
    int                      nLevels_;
    int                      initThFAST_, minThFAST_;
    int                      _dbow_version{2};
    keypoint_detector_type   _keypoint_detector;
    keypoint_descriptor_type _keypoint_descriptor{keypoint_descriptor_type::orb};
    descriptor_matcher_type  _descriptor_matcher{descriptor_matcher_type::brute_force};
    bool                     _use_optical_flow{false};

    /*
     * Other stuff
     */
    float thFarPoints_;
};
}; // namespace ORB_SLAM3

#endif // ORB_SLAM3_SETTINGS_H
