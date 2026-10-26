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

#ifndef DESCRIPTOR_MATCHER_H
#define DESCRIPTOR_MATCHER_H

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "descriptor_matcher_type.h"
#include "sophus/sim3.hpp"

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

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int search_by_projection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th = 3, const bool bFarPoints = false, const float thFarPoints = 50.0f);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int search_by_projection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono, GeometricCamera *camera);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int search_by_projection(Frame &CurrentFrame, KeyFrame *pKF, const std::set<MapPoint *> &sAlreadyFound, const float th, const int descriptor_threshold, GeometricCamera *camera);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int search_by_projection(KeyFrame *pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpMatched, int th, float distance_multiplier = 1.0);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in Place Recognition (Loop Closing and Merging)
    int search_by_projection(KeyFrame *pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint *> &vpPoints, const std::vector<KeyFrame *> &vpPointsKFs, std::vector<MapPoint *> &vpMatched,
                             std::vector<KeyFrame *> &vpMatchedKF, int th, float distance_multiplier = 1.0);

    // Match selected tracking descriptors constrained by place-recognition vocabulary nodes.
    // Used in Relocalisation and Loop Detection
    int search_by_vocabulary(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);
    int search_by_vocabulary(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int search_for_initialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int search_for_triangulation(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse = false);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int search_by_sim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th = 3.0, const bool bRight = false);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const std::vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

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

#endif // DESCRIPTOR_MATCHER_H