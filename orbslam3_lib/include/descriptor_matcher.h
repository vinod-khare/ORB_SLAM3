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

#include "descriptor_matcher_type.h"
#include "frame.h"
#include "key_frame.h"
#include "map_point.h"
#include "sophus/sim3.hpp"

#include <functional>
#include <memory>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace ORB_SLAM3
{

class descriptor_matcher
{
  public:
    descriptor_matcher(float nnratio = 0.6, bool checkOri = true);

    // Distance in 32-byte Hamming-equivalent acceptance units.
    static float                          descriptor_distance(const cv::Mat &first, const cv::Mat &second);
    static void                           set_backend(descriptor_matcher_type backend);
    static cv::Ptr<cv::DescriptorMatcher> create(const cv::Mat &descriptors);
    static void                           knn_match(const cv::Mat &query, const cv::Mat &train, std::vector<std::vector<cv::DMatch>> &matches, int neighbors = 2);
    static std::vector<size_t>            rank_candidates(const cv::Mat &query, const cv::Mat &train, const std::vector<size_t> &indices);
    static std::vector<size_t>            rank_candidates(const cv::Mat &query, const std::vector<size_t> &indices, const std::function<const cv::Mat &(size_t)> &get_descriptor);
    static std::vector<size_t>            rank_candidates(const cv::Mat &query, const std::vector<keypoint> &train, const std::vector<size_t> &indices);
    static std::vector<size_t>            rank_candidates(const cv::Mat &query, const frame &train, const std::vector<size_t> &indices, bool right = false);
    static std::vector<size_t>            rank_candidates(const cv::Mat &query, const key_frame &train, const std::vector<size_t> &indices, bool right = false);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int search_by_projection(frame &F, const std::vector<map_point *> &vpMapPoints, const float th = 3, const bool bFarPoints = false, const float thFarPoints = 50.0f);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int search_by_projection(frame &CurrentFrame, const frame &LastFrame, const float th, const bool bMono, const camera_vector &cameras);

    // Project MapPoints seen in key_frame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int search_by_projection(frame &CurrentFrame, key_frame *pKF, const std::set<map_point *> &sAlreadyFound, const float th, const int descriptor_threshold, const camera_vector &cameras);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int search_by_projection(
        camera_view cameras, key_frame *pKF, Sophus::Sim3<float> &Scw, const std::vector<map_point *> &vpPoints, std::vector<map_point *> &vpMatched, int th, float distance_multiplier = 1.0);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in Place Recognition (Loop Closing and Merging)
    int search_by_projection(camera_view                     cameras,
                             key_frame                      *pKF,
                             Sophus::Sim3<float>            &Scw,
                             const std::vector<map_point *> &vpPoints,
                             const std::vector<key_frame *> &vpPointsKFs,
                             std::vector<map_point *>       &vpMatched,
                             std::vector<key_frame *>       &vpMatchedKF,
                             int                             th,
                             float                           distance_multiplier = 1.0);

    // Match selected tracking descriptors constrained by place-recognition vocabulary nodes.
    // Used in Relocalisation and Loop Detection
    int search_by_vocabulary(key_frame *pKF, frame &F, std::vector<map_point *> &vpMapPointMatches);
    int search_by_vocabulary(key_frame *pKF1, key_frame *pKF2, std::vector<map_point *> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int search_for_initialization(frame &F1, frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int search_for_triangulation(camera_view cameras, key_frame *pKF1, key_frame *pKF2, std::vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse = false);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int search_by_sim3(camera_view cameras, key_frame *pKF1, key_frame *pKF2, std::vector<map_point *> &vpMatches12, const Sophus::Sim3f &S12, const float th);

    // Project MapPoints into key_frame and search for duplicated MapPoints.
    int fuse(camera_view cameras, key_frame *pKF, const vector<map_point *> &vpMapPoints, const float th = 3.0, const bool bRight = false);

    // Project MapPoints into key_frame using a given Sim3 and search for duplicated MapPoints.
    int fuse(camera_view cameras, key_frame *pKF, Sophus::Sim3f &Scw, const std::vector<map_point *> &vpPoints, float th, vector<map_point *> &vpReplacePoint);

  public:
    static const int low_distance_threshold;
    static const int high_distance_threshold;
    static const int orientation_histogram_bins;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  protected:
    float radius_by_viewing_cos(const float &viewCos);
    void  compute_three_maxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

    float _nearest_neighbor_ratio;
    bool  _check_orientation;
};

} // namespace ORB_SLAM3
