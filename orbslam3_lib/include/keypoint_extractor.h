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

#include "keypoint.h"
#include "keypoint_descriptor_type.h"
#include "keypoint_detector_type.h"

#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

class keypoint_extractor
{
  public:
    keypoint_extractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST, keypoint_detector_type detector_type = keypoint_detector_type::fast,
                       keypoint_descriptor_type descriptor_type = keypoint_descriptor_type::orb);

    ~keypoint_extractor() = default;

    // Compute spatially distributed keypoints and selected OpenCV descriptors.
    // Mask is ignored in the current implementation.
    int operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<keypoint> &_keypoints, cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    int inline get_levels() { return nlevels; }

    float inline get_scale_factor() { return scaleFactor; }

    std::vector<float> inline get_scale_factors() { return mvScaleFactor; }

    std::vector<float> inline get_inverse_scale_factors() { return mvInvScaleFactor; }

    std::vector<float> inline get_scale_sigma_squares() { return mvLevelSigma2; }

    std::vector<float> inline get_inverse_scale_sigma_squares() { return mvInvLevelSigma2; }

    keypoint_detector_type get_detector_type() const { return _detector_type; }

    const cv::Mat &get_place_descriptors() const { return _place_descriptors; }

    std::vector<cv::Mat> mvImagePyramid;

  protected:
    void                      compute_pyramid(cv::Mat image);
    void                      compute_keypoints(std::vector<std::vector<cv::KeyPoint>> &allKeypoints, std::vector<std::vector<cv::KeyPoint>> &display_keypoints);
    std::vector<cv::KeyPoint> distribute_keypoints(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX, const int &maxX, const int &minY, const int &maxY, const int &nFeatures,
                                                   const int &level);

    int    nfeatures;
    double scaleFactor;
    int    nlevels;
    int    iniThFAST;
    int    minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

  private:
    void detect_corners_in_cell(const cv::Mat &cell, std::vector<cv::KeyPoint> &keypoints, int threshold) const;
    void detect_keypoints(int level, int min_x, int max_x, int min_y, int max_y, std::vector<cv::KeyPoint> &keypoints);

    keypoint_detector_type   _detector_type;
    cv::Ptr<cv::Feature2D>   _detector;
    keypoint_descriptor_type _descriptor_type;
    cv::Ptr<cv::Feature2D>   _descriptor;
    cv::Ptr<cv::Feature2D>   _place_descriptor;
    cv::Mat                  _place_descriptors;
};

} // namespace ORB_SLAM3
