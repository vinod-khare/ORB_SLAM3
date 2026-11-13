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

#include "descriptor_matcher.h"

#include <limits.h>

#include <atomic>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/flann.hpp>

#include <DBoW2/FeatureVector.h>
#include <stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

const int descriptor_matcher::high_distance_threshold    = 100;
const int descriptor_matcher::low_distance_threshold     = 50;
const int descriptor_matcher::orientation_histogram_bins = 30;

descriptor_matcher::descriptor_matcher(float nnratio, bool checkOri) : _nearest_neighbor_ratio(nnratio), _check_orientation(checkOri) {}

namespace
{
std::atomic<descriptor_matcher_type> selected_backend{descriptor_matcher_type::brute_force};
}

void descriptor_matcher::set_backend(descriptor_matcher_type backend) { selected_backend.store(backend); }

cv::Ptr<cv::DescriptorMatcher> descriptor_matcher::create(const cv::Mat &descriptors)
{
    if (selected_backend.load() == descriptor_matcher_type::brute_force)
    {
        return cv::BFMatcher::create(descriptors.depth() == CV_8U ? cv::NORM_HAMMING : cv::NORM_L2);
    }
    cv::Ptr<cv::flann::IndexParams> index_params;
    if (descriptors.depth() == CV_8U)
    {
        index_params = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
    }
    else
    {
        index_params = cv::makePtr<cv::flann::KDTreeIndexParams>(4);
    }
    return cv::makePtr<cv::FlannBasedMatcher>(index_params, cv::makePtr<cv::flann::SearchParams>(64));
}

void descriptor_matcher::knn_match(const cv::Mat &query, const cv::Mat &train, std::vector<std::vector<cv::DMatch>> &matches, int neighbors)
{
    matches.clear();
    if (query.empty() || train.empty())
    {
        return;
    }
    CV_Assert(query.type() == train.type() && query.cols == train.cols && neighbors > 0);
    const int                      count = std::min(neighbors, train.rows);
    cv::Ptr<cv::DescriptorMatcher> matcher;
    if (count == train.rows)
    {
        matcher = cv::BFMatcher::create(train.depth() == CV_8U ? cv::NORM_HAMMING : cv::NORM_L2);
    }
    else
    {
        matcher = create(train);
    }
    matcher->knnMatch(query, train, matches, count);
}

std::vector<size_t> descriptor_matcher::rank_candidates(const cv::Mat &query, const cv::Mat &train, const std::vector<size_t> &indices)
{
    cv::Mat candidates;
    for (const auto index : indices)
    {
        candidates.push_back(train.row(static_cast<int>(index)));
    }
    std::vector<std::vector<cv::DMatch>> matches;
    knn_match(query, candidates, matches, std::max(1, static_cast<int>(indices.size())));
    std::vector<size_t> result;
    if (!matches.empty())
    {
        for (const auto &match : matches.front())
        {
            result.push_back(indices[match.trainIdx]);
        }
    }
    return result;
}

std::vector<size_t> descriptor_matcher::rank_candidates(const cv::Mat &query, const std::vector<size_t> &indices, const std::function<const cv::Mat &(size_t)> &get_descriptor)
{
    cv::Mat candidates;
    for (const auto index : indices)
    {
        candidates.push_back(get_descriptor(index));
    }
    std::vector<std::vector<cv::DMatch>> matches;
    knn_match(query, candidates, matches, std::max(1, static_cast<int>(indices.size())));
    std::vector<size_t> result;
    if (!matches.empty())
    {
        for (const auto &match : matches.front())
        {
            result.push_back(indices[match.trainIdx]);
        }
    }
    return result;
}

std::vector<size_t> descriptor_matcher::rank_candidates(const cv::Mat &query, const std::vector<keypoint> &train, const std::vector<size_t> &indices)
{
    return rank_candidates(query, indices, [&train](size_t index) -> const cv::Mat & { return train[index].descriptor; });
}

std::vector<size_t> descriptor_matcher::rank_candidates(const cv::Mat &query, const frame &train, const std::vector<size_t> &indices, bool right)
{
    return rank_candidates(query, indices, [&train, right](size_t index) -> const cv::Mat & { return right ? train.right_keypoints[index].descriptor : train.descriptor(index); });
}

std::vector<size_t> descriptor_matcher::rank_candidates(const cv::Mat &query, const key_frame &train, const std::vector<size_t> &indices, bool right)
{
    return rank_candidates(query, indices, [&train, right](size_t index) -> const cv::Mat & { return right ? train.right_keypoints[index].descriptor : train.descriptor(index); });
}

int descriptor_matcher::search_by_projection(frame &F, const vector<map_point *> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
    int nmatches = 0, left = 0, right = 0;

    const bool bFactor = th != 1.0;

    for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
    {
        map_point *pMP = vpMapPoints[iMP];
        if (!pMP->track_in_view && !pMP->track_in_view_right)
        {
            continue;
        }

        if (bFarPoints && pMP->track_depth > thFarPoints)
        {
            continue;
        }

        if (pMP->is_bad())
        {
            continue;
        }

        if (pMP->track_in_view)
        {
            const int &nPredictedLevel = pMP->track_scale_level;

            // The size of the window will depend on the viewing direction
            float r = radius_by_viewing_cos(pMP->track_view_cos);

            if (bFactor)
            {
                r *= th;
            }

            const auto vIndices = rank_candidates(
                pMP->get_descriptor(), F, F.get_features_in_area(pMP->track_projection_x, pMP->track_projection_y, r * F.scale_factors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel));

            if (!vIndices.empty())
            {
                const cv::Mat MPdescriptor = pMP->get_descriptor();

                float bestDist   = 256;
                int   bestLevel  = -1;
                float bestDist2  = 256;
                int   bestLevel2 = -1;
                int   bestIdx    = -1;

                // Get best and second matches with near keypoints
                for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
                {
                    const size_t idx = *vit;

                    if (F.map_points[idx])
                    {
                        if (F.map_points[idx]->get_observation_count() > 0)
                        {
                            continue;
                        }
                    }

                    if (!F.is_fisheye_stereo && F.right_coordinates[idx] > 0)
                    {
                        const float er = fabs(pMP->track_projection_x_right - F.right_coordinates[idx]);
                        if (er > r * F.scale_factors[nPredictedLevel])
                        {
                            continue;
                        }
                    }

                    const cv::Mat &d = F.descriptor(idx);

                    const float dist = descriptor_distance(MPdescriptor, d);

                    if (dist < bestDist)
                    {
                        bestDist2  = bestDist;
                        bestDist   = dist;
                        bestLevel2 = bestLevel;
                        bestLevel  = (!F.is_fisheye_stereo || std::cmp_less(idx, F.keypoints.size())) ? F.keypoints[idx].octave : F.right_keypoints[idx - F.keypoints.size()].octave;
                        bestIdx    = idx;
                    }
                    else if (dist < bestDist2)
                    {
                        bestLevel2 = (!F.is_fisheye_stereo || std::cmp_less(idx, F.keypoints.size())) ? F.keypoints[idx].octave : F.right_keypoints[idx - F.keypoints.size()].octave;
                        bestDist2  = dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if (bestDist <= high_distance_threshold)
                {
                    if (bestLevel == bestLevel2 && bestDist > _nearest_neighbor_ratio * bestDist2)
                    {
                        continue;
                    }

                    if (bestLevel != bestLevel2 || bestDist <= _nearest_neighbor_ratio * bestDist2)
                    {
                        F.map_points[bestIdx] = pMP;

                        if (F.is_fisheye_stereo && F.left_to_right_matches[bestIdx] != -1)
                        { // Also match with the stereo observation at right camera
                            F.map_points[F.left_to_right_matches[bestIdx] + F.keypoints.size()] = pMP;
                            nmatches++;
                            right++;
                        }

                        nmatches++;
                        left++;
                    }
                }
            }
        }

        if (F.is_fisheye_stereo && pMP->track_in_view_right)
        {
            const int &nPredictedLevel = pMP->track_scale_level_right;
            if (nPredictedLevel != -1)
            {
                float r = radius_by_viewing_cos(pMP->track_view_cos_right);

                const auto vIndices = rank_candidates(
                    pMP->get_descriptor(),
                    F,
                    F.get_features_in_area(pMP->track_projection_x_right, pMP->track_projection_y_right, r * F.scale_factors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel, true),
                    true);

                if (vIndices.empty())
                {
                    continue;
                }

                const cv::Mat MPdescriptor = pMP->get_descriptor();

                float bestDist   = 256;
                int   bestLevel  = -1;
                float bestDist2  = 256;
                int   bestLevel2 = -1;
                int   bestIdx    = -1;

                // Get best and second matches with near keypoints
                for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
                {
                    const size_t idx = *vit;

                    if (F.map_points[idx + F.keypoints.size()])
                    {
                        if (F.map_points[idx + F.keypoints.size()]->get_observation_count() > 0)
                        {
                            continue;
                        }
                    }

                    const cv::Mat &d = F.descriptor(idx + F.keypoints.size());

                    const float dist = descriptor_distance(MPdescriptor, d);

                    if (dist < bestDist)
                    {
                        bestDist2  = bestDist;
                        bestDist   = dist;
                        bestLevel2 = bestLevel;
                        bestLevel  = F.right_keypoints[idx].octave;
                        bestIdx    = idx;
                    }
                    else if (dist < bestDist2)
                    {
                        bestLevel2 = F.right_keypoints[idx].octave;
                        bestDist2  = dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if (bestDist <= high_distance_threshold)
                {
                    if (bestLevel == bestLevel2 && bestDist > _nearest_neighbor_ratio * bestDist2)
                    {
                        continue;
                    }

                    if (F.is_fisheye_stereo && F.right_to_left_matches[bestIdx] != -1)
                    { // Also match with the stereo observation at right camera
                        F.map_points[F.right_to_left_matches[bestIdx]] = pMP;
                        nmatches++;
                        left++;
                    }

                    F.map_points[bestIdx + F.keypoints.size()] = pMP;
                    nmatches++;
                    right++;
                }
            }
        }
    }
    return nmatches;
}

float descriptor_matcher::radius_by_viewing_cos(const float &viewCos)
{
    if (viewCos > 0.998)
    {
        return 2.5;
    }
    else
    {
        return 4.0;
    }
}

int descriptor_matcher::search_by_vocabulary(key_frame *pKF, frame &F, vector<map_point *> &vpMapPointMatches)
{
    const vector<map_point *> vpMapPointsKF = pKF->get_map_point_matches();

    vpMapPointMatches = vector<map_point *>(F.feature_count, static_cast<map_point *>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->feature_vector;

    int nmatches = 0;

    vector<int> rotHist[orientation_histogram_bins];
    for (int i = 0; i < orientation_histogram_bins; i++)
    {
        rotHist[i].reserve(500);
    }
    const float factor = 1.0f / orientation_histogram_bins;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit  = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit   = F.feature_vector.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend  = F.feature_vector.end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF  = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                map_point *pMP = vpMapPointsKF[realIdxKF];

                if (!pMP)
                {
                    continue;
                }

                if (pMP->is_bad())
                {
                    continue;
                }

                const cv::Mat &dKF = pKF->descriptor(realIdxKF);

                float bestDist1 = 256;
                int   bestIdxF  = -1;
                float bestDist2 = 256;

                float bestDist1R = 256;
                int   bestIdxFR  = -1;
                float bestDist2R = 256;

                const auto ranked_indices = rank_candidates(dKF, F, std::vector<size_t>(vIndicesF.begin(), vIndicesF.end()));
                for (size_t iF = 0; iF < ranked_indices.size(); iF++)
                {
                    if (!F.is_fisheye_stereo)
                    {
                        const auto realIdxF = ranked_indices[iF];

                        if (vpMapPointMatches[realIdxF])
                        {
                            continue;
                        }

                        const cv::Mat &dF = F.descriptor(realIdxF);

                        const float dist = descriptor_distance(dKF, dF);

                        if (dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF  = realIdxF;
                        }
                        else if (dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }
                    }
                    else
                    {
                        const auto realIdxF = ranked_indices[iF];

                        if (vpMapPointMatches[realIdxF])
                        {
                            continue;
                        }

                        const cv::Mat &dF = F.descriptor(realIdxF);

                        const float dist = descriptor_distance(dKF, dF);

                        if (std::cmp_less(realIdxF, F.keypoints.size()) && dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF  = realIdxF;
                        }
                        else if (std::cmp_less(realIdxF, F.keypoints.size()) && dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }

                        if (std::cmp_greater_equal(realIdxF, F.keypoints.size()) && dist < bestDist1R)
                        {
                            bestDist2R = bestDist1R;
                            bestDist1R = dist;
                            bestIdxFR  = realIdxF;
                        }
                        else if (std::cmp_greater_equal(realIdxF, F.keypoints.size()) && dist < bestDist2R)
                        {
                            bestDist2R = dist;
                        }
                    }
                }

                if (bestDist1 <= low_distance_threshold)
                {
                    if (static_cast<float>(bestDist1) < _nearest_neighbor_ratio * static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF] = pMP;

                        const keypoint &kp = ((!is_fisheye_stereo(pKF->cameras))                           ? pKF->keypoints[realIdxKF]
                                              : (std::cmp_greater_equal(realIdxKF, pKF->keypoints.size())) ? pKF->right_keypoints[realIdxKF - pKF->keypoints.size()]
                                                                                                           : pKF->keypoints[realIdxKF]);

                        if (_check_orientation)
                        {
                            const keypoint &Fkp = (!is_fisheye_stereo(pKF->cameras) || !F.is_fisheye_stereo) ? F.keypoints[bestIdxF]
                                                  : std::cmp_greater_equal(bestIdxF, F.keypoints.size())     ? F.right_keypoints[bestIdxF - F.keypoints.size()]
                                                                                                             : F.keypoints[bestIdxF];

                            float rot = kp.tracking_angle - Fkp.tracking_angle;
                            if (rot < 0.0)
                            {
                                rot += 360.0f;
                            }
                            int bin = round(rot * factor);
                            if (bin == orientation_histogram_bins)
                            {
                                bin = 0;
                            }
                            assert(bin >= 0 && bin < orientation_histogram_bins);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }

                    if (bestDist1R <= low_distance_threshold)
                    {
                        if (static_cast<float>(bestDist1R) < _nearest_neighbor_ratio * static_cast<float>(bestDist2R) || true)
                        {
                            vpMapPointMatches[bestIdxFR] = pMP;

                            const keypoint &kp = ((!is_fisheye_stereo(pKF->cameras))                           ? pKF->keypoints[realIdxKF]
                                                  : (std::cmp_greater_equal(realIdxKF, pKF->keypoints.size())) ? pKF->right_keypoints[realIdxKF - pKF->keypoints.size()]
                                                                                                               : pKF->keypoints[realIdxKF]);

                            if (_check_orientation)
                            {
                                const keypoint &Fkp = (!F.is_fisheye_stereo)                                  ? F.keypoints[bestIdxFR]
                                                      : std::cmp_greater_equal(bestIdxFR, F.keypoints.size()) ? F.right_keypoints[bestIdxFR - F.keypoints.size()]
                                                                                                              : F.keypoints[bestIdxFR];

                                float rot = kp.tracking_angle - Fkp.tracking_angle;
                                if (rot < 0.0)
                                {
                                    rot += 360.0f;
                                }
                                int bin = round(rot * factor);
                                if (bin == orientation_histogram_bins)
                                {
                                    bin = 0;
                                }
                                assert(bin >= 0 && bin < orientation_histogram_bins);
                                rotHist[bin].push_back(bestIdxFR);
                            }
                            nmatches++;
                        }
                    }
                }
            }

            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.feature_vector.lower_bound(KFit->first);
        }
    }

    if (_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        compute_three_maxima(rotHist, orientation_histogram_bins, ind1, ind2, ind3);

        for (int i = 0; i < orientation_histogram_bins; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
            {
                continue;
            }
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]] = static_cast<map_point *>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int descriptor_matcher::search_by_projection(key_frame *pKF, Sophus::Sim3f &Scw, const vector<map_point *> &vpPoints, vector<map_point *> &vpMatched, int th, float distance_multiplier)
{
    // Get Calibration Parameters for later projection
    [[maybe_unused]] const float &fx = pKF->fx;
    [[maybe_unused]] const float &fy = pKF->fy;
    [[maybe_unused]] const float &cx = pKF->cx;
    [[maybe_unused]] const float &cy = pKF->cy;

    Sophus::SE3f    Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
    Eigen::Vector3f Ow  = Tcw.inverse().translation();

    // Set of MapPoints already found in the key_frame
    set<map_point *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<map_point *>(NULL));

    int nmatches = 0;

    // For each Candidate map_point Project and Match
    for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
    {
        map_point *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->is_bad() || spAlreadyFound.count(pMP))
        {
            continue;
        }

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->get_world_position();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0)
        {
            continue;
        }

        // Project into Image
        const Eigen::Vector2f uv = pKF->cameras[0]->project(p3Dc);

        // Point must be inside the image
        if (!pKF->is_in_image(uv(0), uv(1)))
        {
            continue;
        }

        // Depth must be inside the scale invariance region of the point
        const float     maxDistance = pMP->get_max_distance_invariance();
        const float     minDistance = pMP->get_min_distance_invariance();
        Eigen::Vector3f PO          = p3Dw - Ow;
        const float     dist        = PO.norm();

        if (dist < minDistance || dist > maxDistance)
        {
            continue;
        }

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->get_normal();

        if (PO.dot(Pn) < 0.5 * dist)
        {
            continue;
        }

        int nPredictedLevel = pMP->predict_scale(dist, pKF);

        // Search in a radius
        const float radius = th * pKF->scale_factors[nPredictedLevel];

        const auto vIndices = rank_candidates(pMP->get_descriptor(), *pKF, pKF->get_features_in_area(uv(0), uv(1), radius));

        if (vIndices.empty())
        {
            continue;
        }

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->get_descriptor();

        float bestDist = 256;
        int   bestIdx  = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            if (vpMatched[idx])
            {
                continue;
            }

            const int &kpLevel = pKF->keypoints[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
            {
                continue;
            }

            const cv::Mat &dKF = pKF->descriptor(idx);

            const float dist = descriptor_distance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx  = idx;
            }
        }

        if (bestDist <= low_distance_threshold * distance_multiplier)
        {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

int descriptor_matcher::search_by_projection(key_frame                      *pKF,
                                             Sophus::Sim3<float>            &Scw,
                                             const std::vector<map_point *> &vpPoints,
                                             const std::vector<key_frame *> &vpPointsKFs,
                                             std::vector<map_point *>       &vpMatched,
                                             std::vector<key_frame *>       &vpMatchedKF,
                                             int                             th,
                                             float                           distance_multiplier)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f    Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
    Eigen::Vector3f Ow  = Tcw.inverse().translation();

    // Set of MapPoints already found in the key_frame
    set<map_point *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<map_point *>(NULL));

    int nmatches = 0;

    // For each Candidate map_point Project and Match
    for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
    {
        map_point *pMP  = vpPoints[iMP];
        key_frame *pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->is_bad() || spAlreadyFound.count(pMP))
        {
            continue;
        }

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->get_world_position();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0)
        {
            continue;
        }

        // Project into Image
        const float invz = 1 / p3Dc(2);
        const float x    = p3Dc(0) * invz;
        const float y    = p3Dc(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->is_in_image(u, v))
        {
            continue;
        }

        // Depth must be inside the scale invariance region of the point
        const float     maxDistance = pMP->get_max_distance_invariance();
        const float     minDistance = pMP->get_min_distance_invariance();
        Eigen::Vector3f PO          = p3Dw - Ow;
        const float     dist        = PO.norm();

        if (dist < minDistance || dist > maxDistance)
        {
            continue;
        }

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->get_normal();

        if (PO.dot(Pn) < 0.5 * dist)
        {
            continue;
        }

        int nPredictedLevel = pMP->predict_scale(dist, pKF);

        // Search in a radius
        const float radius = th * pKF->scale_factors[nPredictedLevel];

        const auto vIndices = rank_candidates(pMP->get_descriptor(), *pKF, pKF->get_features_in_area(u, v, radius));

        if (vIndices.empty())
        {
            continue;
        }

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->get_descriptor();

        float bestDist = 256;
        int   bestIdx  = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            if (vpMatched[idx])
            {
                continue;
            }

            const int &kpLevel = pKF->keypoints[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
            {
                continue;
            }

            const cv::Mat &dKF = pKF->descriptor(idx);

            const float dist = descriptor_distance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx  = idx;
            }
        }

        if (bestDist <= low_distance_threshold * distance_multiplier)
        {
            vpMatched[bestIdx]   = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }
    }

    return nmatches;
}

int descriptor_matcher::search_for_initialization(frame &F1, frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches = 0;
    vnMatches12  = vector<int>(F1.keypoints.size(), -1);

    vector<int> rotHist[orientation_histogram_bins];
    for (int i = 0; i < orientation_histogram_bins; i++)
    {
        rotHist[i].reserve(500);
    }
    const float factor = 1.0f / orientation_histogram_bins;

    vector<float> vMatchedDistance(F2.keypoints.size(), INT_MAX);
    vector<int>   vnMatches21(F2.keypoints.size(), -1);

    for (size_t i1 = 0, iend1 = F1.keypoints.size(); i1 < iend1; i1++)
    {
        const auto &kp1    = F1.keypoints[i1];
        int         level1 = kp1.octave;
        if (level1 > 0)
        {
            continue;
        }

        auto vIndices2 = rank_candidates(F1.keypoints[i1].descriptor, F2, F2.get_features_in_area(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1));

        if (vIndices2.empty())
        {
            continue;
        }

        const cv::Mat &d1 = F1.keypoints[i1].descriptor;

        float bestDist  = INT_MAX;
        float bestDist2 = INT_MAX;
        int   bestIdx2  = -1;

        for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            const cv::Mat &d2 = F2.keypoints[i2].descriptor;

            float dist = descriptor_distance(d1, d2);

            if (vMatchedDistance[i2] <= dist)
            {
                continue;
            }

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist  = dist;
                bestIdx2  = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist <= low_distance_threshold)
        {
            if (bestDist < (float)bestDist2 * _nearest_neighbor_ratio)
            {
                if (vnMatches21[bestIdx2] >= 0)
                {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1]            = bestIdx2;
                vnMatches21[bestIdx2]      = i1;
                vMatchedDistance[bestIdx2] = bestDist;
                nmatches++;

                if (_check_orientation)
                {
                    float rot = F1.keypoints[i1].tracking_angle - F2.keypoints[bestIdx2].tracking_angle;
                    if (rot < 0.0)
                    {
                        rot += 360.0f;
                    }
                    int bin = round(rot * factor);
                    if (bin == orientation_histogram_bins)
                    {
                        bin = 0;
                    }
                    assert(bin >= 0 && bin < orientation_histogram_bins);
                    rotHist[bin].push_back(i1);
                }
            }
        }
    }

    if (_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        compute_three_maxima(rotHist, orientation_histogram_bins, ind1, ind2, ind3);

        for (int i = 0; i < orientation_histogram_bins; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
            {
                continue;
            }
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0)
                {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    // Update prev matched
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
    {
        if (vnMatches12[i1] >= 0)
        {
            vbPrevMatched[i1] = F2.keypoints[vnMatches12[i1]].pt_undistorted;
        }
    }

    return nmatches;
}

int descriptor_matcher::search_by_vocabulary(key_frame *pKF1, key_frame *pKF2, vector<map_point *> &vpMatches12)
{
    const vector<keypoint>     &vKeysUn1     = pKF1->keypoints;
    const DBoW2::FeatureVector &vFeatVec1    = pKF1->feature_vector;
    const vector<map_point *>   vpMapPoints1 = pKF1->get_map_point_matches();

    const vector<keypoint>     &vKeysUn2     = pKF2->keypoints;
    const DBoW2::FeatureVector &vFeatVec2    = pKF2->feature_vector;
    const vector<map_point *>   vpMapPoints2 = pKF2->get_map_point_matches();

    vpMatches12 = vector<map_point *>(vpMapPoints1.size(), static_cast<map_point *>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(), false);

    vector<int> rotHist[orientation_histogram_bins];
    for (int i = 0; i < orientation_histogram_bins; i++)
    {
        rotHist[i].reserve(500);
    }

    const float factor = 1.0f / orientation_histogram_bins;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it  = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it  = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                if (is_fisheye_stereo(pKF1->cameras) && idx1 >= pKF1->keypoints.size())
                {
                    continue;
                }

                map_point *pMP1 = vpMapPoints1[idx1];
                if (!pMP1)
                {
                    continue;
                }
                if (pMP1->is_bad())
                {
                    continue;
                }

                const cv::Mat &d1 = pKF1->descriptor(idx1);

                float bestDist1 = 256;
                int   bestIdx2  = -1;
                float bestDist2 = 256;

                const auto ranked_indices = rank_candidates(d1, *pKF2, std::vector<size_t>(f2it->second.begin(), f2it->second.end()));
                for (const auto idx2 : ranked_indices)
                {

                    if (is_fisheye_stereo(pKF2->cameras) && idx2 >= pKF2->keypoints.size())
                    {
                        continue;
                    }

                    map_point *pMP2 = vpMapPoints2[idx2];

                    if (vbMatched2[idx2] || !pMP2)
                    {
                        continue;
                    }

                    if (pMP2->is_bad())
                    {
                        continue;
                    }

                    const cv::Mat &d2 = pKF2->descriptor(idx2);

                    float dist = descriptor_distance(d1, d2);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2  = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 < low_distance_threshold)
                {
                    if (static_cast<float>(bestDist1) < _nearest_neighbor_ratio * static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]    = vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2] = true;

                        if (_check_orientation)
                        {
                            float rot = vKeysUn1[idx1].tracking_angle - vKeysUn2[bestIdx2].tracking_angle;
                            if (rot < 0.0)
                            {
                                rot += 360.0f;
                            }
                            int bin = round(rot * factor);
                            if (bin == orientation_histogram_bins)
                            {
                                bin = 0;
                            }
                            assert(bin >= 0 && bin < orientation_histogram_bins);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        compute_three_maxima(rotHist, orientation_histogram_bins, ind1, ind2, ind3);

        for (int i = 0; i < orientation_histogram_bins; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
            {
                continue;
            }
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMatches12[rotHist[i][j]] = static_cast<map_point *>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int descriptor_matcher::search_for_triangulation(key_frame *pKF1, key_frame *pKF2, vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->feature_vector;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->feature_vector;

    // Compute epipole in second image
    Sophus::SE3f    T1w = pKF1->get_pose();
    Sophus::SE3f    T2w = pKF2->get_pose();
    Sophus::SE3f    Tw2 = pKF2->get_pose_inverse(); // for convenience
    Eigen::Vector3f Cw  = pKF1->get_camera_center();
    Eigen::Vector3f C2  = T2w * Cw;

    Eigen::Vector2f ep = pKF2->cameras[0]->project(C2);
    Sophus::SE3f    T12;
    Sophus::SE3f    Tll, Tlr, Trl, Trr;
    Eigen::Matrix3f R12; // for fastest computation
    Eigen::Vector3f t12; // for fastest computation

    size_t camera_index1 = 0;
    size_t camera_index2 = 0;

    if (!is_fisheye_stereo(pKF1->cameras) && !is_fisheye_stereo(pKF2->cameras))
    {
        T12 = T1w * Tw2;
        R12 = T12.rotationMatrix();
        t12 = T12.translation();
    }
    else
    {
        Sophus::SE3f Tr1w = pKF1->get_right_pose();
        Sophus::SE3f Twr2 = pKF2->get_right_pose_inverse();
        Tll               = T1w * Tw2;
        Tlr               = T1w * Twr2;
        Trl               = Tr1w * Tw2;
        Trr               = Tr1w * Twr2;
    }

    Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr = Tlr.rotationMatrix(), Rrl = Trl.rotationMatrix(), Rrr = Trr.rotationMatrix();
    Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node
    int          nmatches = 0;
    vector<bool> vbMatched2(pKF2->feature_count, false);
    vector<int>  vMatches12(pKF1->feature_count, -1);

    vector<int> rotHist[orientation_histogram_bins];
    for (int i = 0; i < orientation_histogram_bins; i++)
    {
        rotHist[i].reserve(500);
    }

    const float factor = 1.0f / orientation_histogram_bins;

    DBoW2::FeatureVector::const_iterator f1it  = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it  = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                map_point *pMP1 = pKF1->get_map_point(idx1);

                // If there is already a map_point skip
                if (pMP1)
                {
                    continue;
                }

                const bool bStereo1 = (!is_fisheye_stereo(pKF1->cameras) && pKF1->right_coordinates[idx1] >= 0);

                if (bOnlyStereo)
                {
                    if (!bStereo1)
                    {
                        continue;
                    }
                }

                const auto &kp1 = ((!is_fisheye_stereo(pKF1->cameras))             ? pKF1->keypoints[idx1]
                                   : (std::cmp_less(idx1, pKF1->keypoints.size())) ? pKF1->keypoints[idx1]
                                                                                   : pKF1->right_keypoints[idx1 - pKF1->keypoints.size()]);

                const bool bRight1 = is_fisheye_stereo(pKF1->cameras) && std::cmp_greater_equal(idx1, pKF1->keypoints.size());

                const cv::Mat &d1 = pKF1->descriptor(idx1);

                float bestDist = low_distance_threshold;
                int   bestIdx2 = -1;

                const auto ranked_indices = rank_candidates(d1, *pKF2, std::vector<size_t>(f2it->second.begin(), f2it->second.end()));
                for (auto idx2 : ranked_indices)
                {

                    map_point *pMP2 = pKF2->get_map_point(idx2);

                    // If we have already matched or there is a map_point skip
                    if (vbMatched2[idx2] || pMP2)
                    {
                        continue;
                    }

                    const bool bStereo2 = (!is_fisheye_stereo(pKF2->cameras) && pKF2->right_coordinates[idx2] >= 0);

                    if (bOnlyStereo)
                    {
                        if (!bStereo2)
                        {
                            continue;
                        }
                    }

                    const cv::Mat &d2 = pKF2->descriptor(idx2);

                    const float dist = descriptor_distance(d1, d2);

                    if (dist > low_distance_threshold || dist > bestDist)
                    {
                        continue;
                    }

                    const auto &kp2     = ((!is_fisheye_stereo(pKF2->cameras))             ? pKF2->keypoints[idx2]
                                           : (std::cmp_less(idx2, pKF2->keypoints.size())) ? pKF2->keypoints[idx2]
                                                                                           : pKF2->right_keypoints[idx2 - pKF2->keypoints.size()]);
                    const bool  bRight2 = is_fisheye_stereo(pKF2->cameras) && std::cmp_greater_equal(idx2, pKF2->keypoints.size());

                    if (!bStereo1 && !bStereo2 && !is_fisheye_stereo(pKF1->cameras))
                    {
                        const float distex = ep(0) - kp2.pt_undistorted.x;
                        const float distey = ep(1) - kp2.pt_undistorted.y;
                        if (distex * distex + distey * distey < 100 * pKF2->scale_factors[kp2.octave])
                        {
                            continue;
                        }
                    }

                    if (is_fisheye_stereo(pKF1->cameras) && is_fisheye_stereo(pKF2->cameras))
                    {
                        if (bRight1 && bRight2)
                        {
                            R12 = Rrr;
                            t12 = trr;
                            T12 = Trr;

                            camera_index1 = 1;
                            camera_index2 = 1;
                        }
                        else if (bRight1 && !bRight2)
                        {
                            R12 = Rrl;
                            t12 = trl;
                            T12 = Trl;

                            camera_index1 = 1;
                            camera_index2 = 0;
                        }
                        else if (!bRight1 && bRight2)
                        {
                            R12 = Rlr;
                            t12 = tlr;
                            T12 = Tlr;

                            camera_index1 = 0;
                            camera_index2 = 1;
                        }
                        else
                        {
                            R12 = Rll;
                            t12 = tll;
                            T12 = Tll;

                            camera_index1 = 0;
                            camera_index2 = 0;
                        }
                    }

                    auto cv_kp1 = static_cast<cv::KeyPoint>(kp1);
                    auto cv_kp2 = static_cast<cv::KeyPoint>(kp2);
                    cv_kp1.pt   = kp1.pt_undistorted;
                    cv_kp2.pt   = kp2.pt_undistorted;
                    if (bCoarse || pKF1->cameras[camera_index1]->epipolar_constrain(*pKF2->cameras[camera_index2],
                                                                                    cv_kp1,
                                                                                    cv_kp2,
                                                                                    R12,
                                                                                    t12,
                                                                                    pKF1->level_sigma_squared[kp1.octave],
                                                                                    pKF2->level_sigma_squared[kp2.octave])) // MODIFICATION_2
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestIdx2 >= 0)
                {
                    const keypoint &tracking_kp1 = bRight1 ? pKF1->right_keypoints[idx1 - pKF1->keypoints.size()] : pKF1->keypoints[idx1];
                    const keypoint &kp2          = ((!is_fisheye_stereo(pKF2->cameras))               ? pKF2->keypoints[bestIdx2]
                                                    : std::cmp_less(bestIdx2, pKF2->keypoints.size()) ? pKF2->keypoints[bestIdx2]
                                                                                                      : pKF2->right_keypoints[bestIdx2 - pKF2->keypoints.size()]);
                    vMatches12[idx1]             = bestIdx2;
                    nmatches++;

                    if (_check_orientation)
                    {
                        float rot = tracking_kp1.tracking_angle - kp2.tracking_angle;
                        if (rot < 0.0)
                        {
                            rot += 360.0f;
                        }
                        int bin = round(rot * factor);
                        if (bin == orientation_histogram_bins)
                        {
                            bin = 0;
                        }
                        assert(bin >= 0 && bin < orientation_histogram_bins);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        compute_three_maxima(rotHist, orientation_histogram_bins, ind1, ind2, ind3);

        for (int i = 0; i < orientation_histogram_bins; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
            {
                continue;
            }
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0)
        {
            continue;
        }
        vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
    }

    return nmatches;
}

int descriptor_matcher::fuse(key_frame *pKF, const vector<map_point *> &vpMapPoints, const float th, const bool bRight)
{
    auto           &selected_camera = *pKF->cameras[bRight ? 1 : 0];
    Sophus::SE3f    Tcw;
    Eigen::Vector3f Ow;

    if (bRight)
    {
        Tcw = pKF->get_right_pose();
        Ow  = pKF->get_right_camera_center();
    }
    else
    {
        Tcw = pKF->get_pose();
        Ow  = pKF->get_camera_center();
    }

    [[maybe_unused]] const float &fx = pKF->fx;
    [[maybe_unused]] const float &fy = pKF->fy;
    [[maybe_unused]] const float &cx = pKF->cx;
    [[maybe_unused]] const float &cy = pKF->cy;
    const float                  &bf = pKF->mbf;

    int nFused = 0;

    const int nMPs = vpMapPoints.size();

    // For debbuging
    int count_notMP = 0, count_bad = 0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal = 0, count_notidx = 0, count_thcheck = 0;
    for (int i = 0; i < nMPs; i++)
    {
        map_point *pMP = vpMapPoints[i];

        if (!pMP)
        {
            count_notMP++;
            continue;
        }

        if (pMP->is_bad())
        {
            count_bad++;
            continue;
        }
        else if (pMP->is_in_key_frame(pKF))
        {
            count_isinKF++;
            continue;
        }

        Eigen::Vector3f p3Dw = pMP->get_world_position();
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0f)
        {
            count_negdepth++;
            continue;
        }

        const float invz = 1 / p3Dc(2);

        const Eigen::Vector2f uv = selected_camera.project(p3Dc);

        // Point must be inside the image
        if (!pKF->is_in_image(uv(0), uv(1)))
        {
            count_notinim++;
            continue;
        }

        const float ur = uv(0) - bf * invz;

        const float     maxDistance = pMP->get_max_distance_invariance();
        const float     minDistance = pMP->get_min_distance_invariance();
        Eigen::Vector3f PO          = p3Dw - Ow;
        const float     dist3D      = PO.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
        {
            count_dist++;
            continue;
        }

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->get_normal();

        if (PO.dot(Pn) < 0.5 * dist3D)
        {
            count_normal++;
            continue;
        }

        int nPredictedLevel = pMP->predict_scale(dist3D, pKF);

        // Search in a radius
        const float radius = th * pKF->scale_factors[nPredictedLevel];

        const auto vIndices = rank_candidates(pMP->get_descriptor(), *pKF, pKF->get_features_in_area(uv(0), uv(1), radius, bRight), bRight);

        if (vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->get_descriptor();

        float bestDist = 256;
        int   bestIdx  = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            size_t      idx = *vit;
            const auto &kp  = ((!is_fisheye_stereo(pKF->cameras) || !bRight) ? pKF->keypoints[idx] : pKF->right_keypoints[idx]);

            const int &kpLevel = kp.octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
            {
                continue;
            }

            if (pKF->right_coordinates[idx] >= 0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt_undistorted.x;
                const float &kpy = kp.pt_undistorted.y;
                const float &kpr = pKF->right_coordinates[idx];
                const float  ex  = uv(0) - kpx;
                const float  ey  = uv(1) - kpy;
                const float  er  = ur - kpr;
                const float  e2  = ex * ex + ey * ey + er * er;

                if (e2 * pKF->inverse_level_sigma_squared[kpLevel] > 7.8)
                {
                    continue;
                }
            }
            else
            {
                const float &kpx = kp.pt_undistorted.x;
                const float &kpy = kp.pt_undistorted.y;
                const float  ex  = uv(0) - kpx;
                const float  ey  = uv(1) - kpy;
                const float  e2  = ex * ex + ey * ey;

                if (e2 * pKF->inverse_level_sigma_squared[kpLevel] > 5.99)
                {
                    continue;
                }
            }

            if (bRight)
            {
                idx += pKF->keypoints.size();
            }

            const cv::Mat &dKF = pKF->descriptor(bRight ? idx + pKF->keypoints.size() : idx);

            const float dist = descriptor_distance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx  = idx;
            }
        }

        // If there is already a map_point replace otherwise add new measurement
        if (bestDist <= low_distance_threshold)
        {
            map_point *pMPinKF = pKF->get_map_point(bestIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->is_bad())
                {
                    if (pMPinKF->get_observation_count() > pMP->get_observation_count())
                    {
                        pMP->replace(pMPinKF);
                    }
                    else
                    {
                        pMPinKF->replace(pMP);
                    }
                }
            }
            else
            {
                pMP->add_observation(pKF, bestIdx);
                pKF->add_map_point(pMP, bestIdx);
            }
            nFused++;
        }
        else
        {
            count_thcheck++;
        }
    }

    return nFused;
}

int descriptor_matcher::fuse(key_frame *pKF, Sophus::Sim3f &Scw, const vector<map_point *> &vpPoints, float th, vector<map_point *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    [[maybe_unused]] const float &fx = pKF->fx;
    [[maybe_unused]] const float &fy = pKF->fy;
    [[maybe_unused]] const float &cx = pKF->cx;
    [[maybe_unused]] const float &cy = pKF->cy;

    // Decompose Scw
    Sophus::SE3f    Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
    Eigen::Vector3f Ow  = Tcw.inverse().translation();

    // Set of MapPoints already found in the key_frame
    const set<map_point *> spAlreadyFound = pKF->get_map_points();

    int nFused = 0;

    const int nPoints = vpPoints.size();

    // For each candidate map_point project and match
    for (int iMP = 0; iMP < nPoints; iMP++)
    {
        map_point *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->is_bad() || spAlreadyFound.count(pMP))
        {
            continue;
        }

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->get_world_position();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc(2) < 0.0f)
        {
            continue;
        }

        // Project into Image
        const Eigen::Vector2f uv = pKF->cameras[0]->project(p3Dc);

        // Point must be inside the image
        if (!pKF->is_in_image(uv(0), uv(1)))
        {
            continue;
        }

        // Depth must be inside the scale pyramid of the image
        const float     maxDistance = pMP->get_max_distance_invariance();
        const float     minDistance = pMP->get_min_distance_invariance();
        Eigen::Vector3f PO          = p3Dw - Ow;
        const float     dist3D      = PO.norm();

        if (dist3D < minDistance || dist3D > maxDistance)
        {
            continue;
        }

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->get_normal();

        if (PO.dot(Pn) < 0.5 * dist3D)
        {
            continue;
        }

        // Compute predicted scale level
        const int nPredictedLevel = pMP->predict_scale(dist3D, pKF);

        // Search in a radius
        const float radius = th * pKF->scale_factors[nPredictedLevel];

        const auto vIndices = rank_candidates(pMP->get_descriptor(), *pKF, pKF->get_features_in_area(uv(0), uv(1), radius));

        if (vIndices.empty())
        {
            continue;
        }

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->get_descriptor();

        float bestDist = INT_MAX;
        int   bestIdx  = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++)
        {
            const size_t idx     = *vit;
            const int   &kpLevel = pKF->keypoints[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
            {
                continue;
            }

            const cv::Mat &dKF = pKF->descriptor(idx);

            float dist = descriptor_distance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx  = idx;
            }
        }

        // If there is already a map_point replace otherwise add new measurement
        if (bestDist <= low_distance_threshold)
        {
            map_point *pMPinKF = pKF->get_map_point(bestIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->is_bad())
                {
                    vpReplacePoint[iMP] = pMPinKF;
                }
            }
            else
            {
                pMP->add_observation(pKF, bestIdx);
                pKF->add_map_point(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int descriptor_matcher::search_by_sim3(key_frame *pKF1, key_frame *pKF2, std::vector<map_point *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 & 2 from world
    Sophus::SE3f T1w = pKF1->get_pose();
    Sophus::SE3f T2w = pKF2->get_pose();

    // Transformation between cameras
    Sophus::Sim3f S21 = S12.inverse();

    const vector<map_point *> vpMapPoints1 = pKF1->get_map_point_matches();
    const int                 N1           = vpMapPoints1.size();

    const vector<map_point *> vpMapPoints2 = pKF2->get_map_point_matches();
    const int                 N2           = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1, false);
    vector<bool> vbAlreadyMatched2(N2, false);

    for (int i = 0; std::cmp_less(i, N1); i++)
    {
        map_point *pMP = vpMatches12[i];
        if (pMP)
        {
            vbAlreadyMatched1[i] = true;
            int idx2             = get<0>(pMP->get_index_in_key_frame(pKF2));
            if (idx2 >= 0 && idx2 < N2)
            {
                vbAlreadyMatched2[idx2] = true;
            }
        }
    }

    vector<int> vnMatch1(N1, -1);
    vector<int> vnMatch2(N2, -1);

    // Transform from KF1 to KF2 and search
    for (int i1 = 0; i1 < N1; i1++)
    {
        map_point *pMP = vpMapPoints1[i1];

        if (!pMP || vbAlreadyMatched1[i1])
        {
            continue;
        }

        if (pMP->is_bad())
        {
            continue;
        }

        Eigen::Vector3f p3Dw  = pMP->get_world_position();
        Eigen::Vector3f p3Dc1 = T1w * p3Dw;
        Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

        // Depth must be positive
        if (p3Dc2(2) < 0.0)
        {
            continue;
        }

        const float invz = 1.0 / p3Dc2(2);
        const float x    = p3Dc2(0) * invz;
        const float y    = p3Dc2(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF2->is_in_image(u, v))
        {
            continue;
        }

        const float maxDistance = pMP->get_max_distance_invariance();
        const float minDistance = pMP->get_min_distance_invariance();
        const float dist3D      = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if (dist3D < minDistance || dist3D > maxDistance)
        {
            continue;
        }

        // Compute predicted octave
        const int nPredictedLevel = pMP->predict_scale(dist3D, pKF2);

        // Search in a radius
        const float radius = th * pKF2->scale_factors[nPredictedLevel];

        const auto vIndices = rank_candidates(pMP->get_descriptor(), *pKF2, pKF2->get_features_in_area(u, v, radius));

        if (vIndices.empty())
        {
            continue;
        }

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->get_descriptor();

        float bestDist = INT_MAX;
        int   bestIdx  = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const auto &kp = pKF2->keypoints[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
            {
                continue;
            }

            const cv::Mat &dKF = pKF2->descriptor(idx);

            const float dist = descriptor_distance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx  = idx;
            }
        }

        if (bestDist <= high_distance_threshold)
        {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for (int i2 = 0; i2 < N2; i2++)
    {
        map_point *pMP = vpMapPoints2[i2];

        if (!pMP || vbAlreadyMatched2[i2])
        {
            continue;
        }

        if (pMP->is_bad())
        {
            continue;
        }

        Eigen::Vector3f p3Dw  = pMP->get_world_position();
        Eigen::Vector3f p3Dc2 = T2w * p3Dw;
        Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

        // Depth must be positive
        if (p3Dc1(2) < 0.0)
        {
            continue;
        }

        const float invz = 1.0 / p3Dc1(2);
        const float x    = p3Dc1(0) * invz;
        const float y    = p3Dc1(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF1->is_in_image(u, v))
        {
            continue;
        }

        const float maxDistance = pMP->get_max_distance_invariance();
        const float minDistance = pMP->get_min_distance_invariance();
        const float dist3D      = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
        {
            continue;
        }

        // Compute predicted octave
        const int nPredictedLevel = pMP->predict_scale(dist3D, pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th * pKF1->scale_factors[nPredictedLevel];

        const auto vIndices = rank_candidates(pMP->get_descriptor(), *pKF1, pKF1->get_features_in_area(u, v, radius));

        if (vIndices.empty())
        {
            continue;
        }

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->get_descriptor();

        float bestDist = INT_MAX;
        int   bestIdx  = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const auto &kp = pKF1->keypoints[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
            {
                continue;
            }

            const cv::Mat &dKF = pKF1->descriptor(idx);

            const float dist = descriptor_distance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx  = idx;
            }
        }

        if (bestDist <= high_distance_threshold)
        {
            vnMatch2[i2] = bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for (int i1 = 0; i1 < N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if (idx2 >= 0)
        {
            int idx1 = vnMatch2[idx2];
            if (idx1 == i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

int descriptor_matcher::search_by_projection(frame &CurrentFrame, const frame &LastFrame, const float th, const bool bMono, const camera_vector &cameras)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[orientation_histogram_bins];
    for (int i = 0; i < orientation_histogram_bins; i++)
    {
        rotHist[i].reserve(500);
    }
    const float factor = 1.0f / orientation_histogram_bins;

    const Sophus::SE3f    Tcw = CurrentFrame.get_pose();
    const Eigen::Vector3f twc = Tcw.inverse().translation();

    const Sophus::SE3f    Tlw = LastFrame.get_pose();
    const Eigen::Vector3f tlc = Tlw * twc;

    const bool bForward  = tlc(2) > CurrentFrame.baseline && !bMono;
    const bool bBackward = -tlc(2) > CurrentFrame.baseline && !bMono;

    for (int i = 0; i < LastFrame.feature_count; i++)
    {
        map_point *pMP = LastFrame.map_points[i];
        if (pMP)
        {
            if (!LastFrame.outliers[i])
            {
                // Project
                Eigen::Vector3f x3Dw = pMP->get_world_position();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;

                [[maybe_unused]] const float xc    = x3Dc(0);
                [[maybe_unused]] const float yc    = x3Dc(1);
                const float                  invzc = 1.0 / x3Dc(2);

                if (invzc < 0)
                {
                    continue;
                }

                Eigen::Vector2f uv = cameras[0]->project(x3Dc);

                if (uv(0) < CurrentFrame.min_x || uv(0) > CurrentFrame.max_x)
                {
                    continue;
                }
                if (uv(1) < CurrentFrame.min_y || uv(1) > CurrentFrame.max_y)
                {
                    continue;
                }

                int nLastOctave = (!LastFrame.is_fisheye_stereo || std::cmp_less(i, LastFrame.keypoints.size())) ? LastFrame.keypoints[i].octave
                                                                                                                 : LastFrame.right_keypoints[i - LastFrame.keypoints.size()].octave;

                // Search in a window. Size depends on scale
                float radius = th * CurrentFrame.scale_factors[nLastOctave];

                vector<size_t> vIndices2;

                if (bForward)
                {
                    vIndices2 = CurrentFrame.get_features_in_area(uv(0), uv(1), radius, nLastOctave);
                }
                else if (bBackward)
                {
                    vIndices2 = CurrentFrame.get_features_in_area(uv(0), uv(1), radius, 0, nLastOctave);
                }
                else
                {
                    vIndices2 = CurrentFrame.get_features_in_area(uv(0), uv(1), radius, nLastOctave - 1, nLastOctave + 1);
                }

                if (vIndices2.empty())
                {
                    continue;
                }

                const cv::Mat dMP = pMP->get_descriptor();

                vIndices2      = rank_candidates(dMP, CurrentFrame, vIndices2);
                float bestDist = 256;
                int   bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                {
                    const size_t i2 = *vit;

                    if (CurrentFrame.map_points[i2])
                    {
                        if (CurrentFrame.map_points[i2]->get_observation_count() > 0)
                        {
                            continue;
                        }
                    }

                    if (!CurrentFrame.is_fisheye_stereo && CurrentFrame.right_coordinates[i2] > 0)
                    {
                        const float ur = uv(0) - CurrentFrame.baseline_fx * invzc;
                        const float er = fabs(ur - CurrentFrame.right_coordinates[i2]);
                        if (er > radius)
                        {
                            continue;
                        }
                    }

                    const cv::Mat &d = CurrentFrame.descriptor(i2);

                    const float dist = descriptor_distance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= high_distance_threshold)
                {
                    CurrentFrame.map_points[bestIdx2] = pMP;
                    nmatches++;

                    if (_check_orientation)
                    {
                        const keypoint &kpLF = ((!LastFrame.is_fisheye_stereo)                 ? LastFrame.keypoints[i]
                                                : std::cmp_less(i, LastFrame.keypoints.size()) ? LastFrame.keypoints[i]
                                                                                               : LastFrame.right_keypoints[i - LastFrame.keypoints.size()]);

                        const keypoint &kpCF = ((!CurrentFrame.is_fisheye_stereo)                        ? CurrentFrame.keypoints[bestIdx2]
                                                : std::cmp_less(bestIdx2, CurrentFrame.keypoints.size()) ? CurrentFrame.keypoints[bestIdx2]
                                                                                                         : CurrentFrame.right_keypoints[bestIdx2 - CurrentFrame.keypoints.size()]);
                        float           rot  = kpLF.tracking_angle - kpCF.tracking_angle;
                        if (rot < 0.0)
                        {
                            rot += 360.0f;
                        }
                        int bin = round(rot * factor);
                        if (bin == orientation_histogram_bins)
                        {
                            bin = 0;
                        }
                        assert(bin >= 0 && bin < orientation_histogram_bins);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
                if (CurrentFrame.is_fisheye_stereo)
                {
                    Eigen::Vector3f x3Dr = CurrentFrame.get_relative_pose_trl() * x3Dc;
                    Eigen::Vector2f uv   = cameras[0]->project(x3Dr);

                    int nLastOctave = (!LastFrame.is_fisheye_stereo || std::cmp_less(i, LastFrame.keypoints.size())) ? LastFrame.keypoints[i].octave
                                                                                                                     : LastFrame.right_keypoints[i - LastFrame.keypoints.size()].octave;

                    // Search in a window. Size depends on scale
                    float radius = th * CurrentFrame.scale_factors[nLastOctave];

                    vector<size_t> vIndices2;

                    if (bForward)
                    {
                        vIndices2 = CurrentFrame.get_features_in_area(uv(0), uv(1), radius, nLastOctave, -1, true);
                    }
                    else if (bBackward)
                    {
                        vIndices2 = CurrentFrame.get_features_in_area(uv(0), uv(1), radius, 0, nLastOctave, true);
                    }
                    else
                    {
                        vIndices2 = CurrentFrame.get_features_in_area(uv(0), uv(1), radius, nLastOctave - 1, nLastOctave + 1, true);
                    }

                    const cv::Mat dMP = pMP->get_descriptor();

                    vIndices2      = rank_candidates(dMP, CurrentFrame, vIndices2, true);
                    float bestDist = 256;
                    int   bestIdx2 = -1;

                    for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if (CurrentFrame.map_points[i2 + CurrentFrame.keypoints.size()])
                        {
                            if (CurrentFrame.map_points[i2 + CurrentFrame.keypoints.size()]->get_observation_count() > 0)
                            {
                                continue;
                            }
                        }

                        const cv::Mat &d = CurrentFrame.descriptor(i2 + CurrentFrame.keypoints.size());

                        const float dist = descriptor_distance(dMP, d);

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    if (bestDist <= high_distance_threshold)
                    {
                        CurrentFrame.map_points[bestIdx2 + CurrentFrame.keypoints.size()] = pMP;
                        nmatches++;
                        if (_check_orientation)
                        {
                            const keypoint &kpLF = ((!LastFrame.is_fisheye_stereo)                 ? LastFrame.keypoints[i]
                                                    : std::cmp_less(i, LastFrame.keypoints.size()) ? LastFrame.keypoints[i]
                                                                                                   : LastFrame.right_keypoints[i - LastFrame.keypoints.size()]);

                            const keypoint &kpCF = CurrentFrame.right_keypoints[bestIdx2];

                            float rot = kpLF.tracking_angle - kpCF.tracking_angle;
                            if (rot < 0.0)
                            {
                                rot += 360.0f;
                            }
                            int bin = round(rot * factor);
                            if (bin == orientation_histogram_bins)
                            {
                                bin = 0;
                            }
                            assert(bin >= 0 && bin < orientation_histogram_bins);
                            rotHist[bin].push_back(bestIdx2 + CurrentFrame.keypoints.size());
                        }
                    }
                }
            }
        }
    }

    // Apply rotation consistency
    if (_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        compute_three_maxima(rotHist, orientation_histogram_bins, ind1, ind2, ind3);

        for (int i = 0; i < orientation_histogram_bins; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    CurrentFrame.map_points[rotHist[i][j]] = static_cast<map_point *>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int descriptor_matcher::search_by_projection(
    frame &CurrentFrame, key_frame *pKF, const set<map_point *> &sAlreadyFound, const float th, const int descriptor_threshold, const camera_vector &cameras)
{
    int nmatches = 0;

    const Sophus::SE3f Tcw = CurrentFrame.get_pose();
    Eigen::Vector3f    Ow  = Tcw.inverse().translation();

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[orientation_histogram_bins];
    for (int i = 0; i < orientation_histogram_bins; i++)
    {
        rotHist[i].reserve(500);
    }
    const float factor = 1.0f / orientation_histogram_bins;

    const vector<map_point *> vpMPs = pKF->get_map_point_matches();

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        map_point *pMP = vpMPs[i];

        if (pMP)
        {
            if (!pMP->is_bad() && !sAlreadyFound.count(pMP))
            {
                // Project
                Eigen::Vector3f x3Dw = pMP->get_world_position();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;

                const Eigen::Vector2f uv = cameras[0]->project(x3Dc);

                if (uv(0) < CurrentFrame.min_x || uv(0) > CurrentFrame.max_x)
                {
                    continue;
                }
                if (uv(1) < CurrentFrame.min_y || uv(1) > CurrentFrame.max_y)
                {
                    continue;
                }

                // Compute predicted scale level
                Eigen::Vector3f PO     = x3Dw - Ow;
                float           dist3D = PO.norm();

                const float maxDistance = pMP->get_max_distance_invariance();
                const float minDistance = pMP->get_min_distance_invariance();

                // Depth must be inside the scale pyramid of the image
                if (dist3D < minDistance || dist3D > maxDistance)
                {
                    continue;
                }

                int nPredictedLevel = pMP->predict_scale(dist3D, &CurrentFrame);

                // Search in a window
                const float radius = th * CurrentFrame.scale_factors[nPredictedLevel];

                const auto vIndices2 = rank_candidates(pMP->get_descriptor(), CurrentFrame, CurrentFrame.get_features_in_area(uv(0), uv(1), radius, nPredictedLevel - 1, nPredictedLevel + 1));

                if (vIndices2.empty())
                {
                    continue;
                }

                const cv::Mat dMP = pMP->get_descriptor();

                float bestDist = 256;
                int   bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if (CurrentFrame.map_points[i2])
                    {
                        continue;
                    }

                    const cv::Mat &d = CurrentFrame.descriptor(i2);

                    const float dist = descriptor_distance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= descriptor_threshold)
                {
                    CurrentFrame.map_points[bestIdx2] = pMP;
                    nmatches++;

                    if (_check_orientation)
                    {
                        // Fisheye frames index right-camera keypoints after the left ones.
                        const auto angle_of = [](const std::vector<keypoint> &left, const std::vector<keypoint> &right, bool fisheye_stereo, size_t idx)
                        { return (!fisheye_stereo || idx < left.size()) ? left[idx].tracking_angle : right[idx - left.size()].tracking_angle; };
                        float rot = angle_of(pKF->keypoints, pKF->right_keypoints, is_fisheye_stereo(pKF->cameras), i) -
                                    angle_of(CurrentFrame.keypoints, CurrentFrame.right_keypoints, CurrentFrame.is_fisheye_stereo, bestIdx2);
                        if (rot < 0.0)
                        {
                            rot += 360.0f;
                        }
                        int bin = round(rot * factor);
                        if (bin == orientation_histogram_bins)
                        {
                            bin = 0;
                        }
                        assert(bin >= 0 && bin < orientation_histogram_bins);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    if (_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        compute_three_maxima(rotHist, orientation_histogram_bins, ind1, ind2, ind3);

        for (int i = 0; i < orientation_histogram_bins; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    CurrentFrame.map_points[rotHist[i][j]] = NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void descriptor_matcher::compute_three_maxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

float descriptor_matcher::descriptor_distance(const cv::Mat &first, const cv::Mat &second)
{
    CV_Assert(first.type() == second.type() && first.size() == second.size());
    if (first.depth() == CV_8U)
    {
        return static_cast<float>(cv::norm(first, second, cv::NORM_HAMMING) * 32.0 / first.cols);
    }
    return static_cast<float>(cv::norm(first, second, cv::NORM_L2) / 4.0);
}

} // namespace ORB_SLAM3
