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

#include "map_point.h"

#include "descriptor_matcher.h"

#include <mutex>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

long unsigned int map_point::next_id = 0;
mutex             map_point::global_mutex;

map_point::map_point()
    : first_key_frame_id(0), first_frame_id(0), observation_count(0), track_reference_for_frame(0), last_frame_seen(0), local_ba_key_frame_id(0), fuse_candidate_key_frame_id(0),
      loop_point_key_frame_id(0), corrected_by_key_frame_id(0), corrected_reference_key_frame_id(0), global_ba_key_frame_id(0), visible(1), found(1), bad(false),
      replaced(static_cast<map_point *>(NULL))
{
    replaced = static_cast<map_point *>(NULL);
}

map_point::map_point(const Eigen::Vector3f &Pos, KeyFrame *pRefKF, map *pMap)
    : first_key_frame_id(pRefKF->mnId), first_frame_id(pRefKF->mnFrameId), observation_count(0), track_reference_for_frame(0), last_frame_seen(0), local_ba_key_frame_id(0),
      fuse_candidate_key_frame_id(0), loop_point_key_frame_id(0), corrected_by_key_frame_id(0), corrected_reference_key_frame_id(0), global_ba_key_frame_id(0), origin_map_id(pMap->get_id()),
      reference_key_frame(pRefKF), visible(1), found(1), bad(false), replaced(nullptr), minimum_distance(0), maximum_distance(0), owner_map(pMap)
{
    set_world_position(Pos);

    normal_vector.setZero();

    track_in_view_right = false;
    track_in_view       = false;

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(owner_map->point_creation_mutex);
    id = next_id++;
}

map_point::map_point(const double invDepth, cv::Point2f uv_init, KeyFrame *pRefKF, KeyFrame *pHostKF, map *pMap)
    : first_key_frame_id(pRefKF->mnId), first_frame_id(pRefKF->mnFrameId), observation_count(0), track_reference_for_frame(0), last_frame_seen(0), local_ba_key_frame_id(0),
      fuse_candidate_key_frame_id(0), loop_point_key_frame_id(0), corrected_by_key_frame_id(0), corrected_reference_key_frame_id(0), global_ba_key_frame_id(0), origin_map_id(pMap->get_id()),
      reference_key_frame(pRefKF), visible(1), found(1), bad(false), replaced(nullptr), minimum_distance(0), maximum_distance(0), owner_map(pMap)
{
    inverse_depth  = invDepth;
    initial_u      = (double)uv_init.x;
    initial_v      = (double)uv_init.y;
    host_key_frame = pHostKF;

    normal_vector.setZero();

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(owner_map->point_creation_mutex);
    id = next_id++;
}

map_point::map_point(const Eigen::Vector3f &Pos, map *pMap, frame *pFrame, const int &idxF)
    : first_key_frame_id(-1), first_frame_id(pFrame->id), observation_count(0), track_reference_for_frame(0), last_frame_seen(0), local_ba_key_frame_id(0), fuse_candidate_key_frame_id(0),
      loop_point_key_frame_id(0), corrected_by_key_frame_id(0), corrected_reference_key_frame_id(0), global_ba_key_frame_id(0), origin_map_id(pMap->get_id()), reference_key_frame(nullptr),
      visible(1), found(1), bad(false), replaced(nullptr), owner_map(pMap)
{
    set_world_position(Pos);

    Eigen::Vector3f Ow;
    if (!pFrame->is_fisheye_stereo || std::cmp_less(idxF, pFrame->keypoints.size()))
    {
        Ow = pFrame->get_camera_center();
    }
    else
    {
        Eigen::Matrix3f Rwl = pFrame->get_rwc();
        Eigen::Vector3f tlr = pFrame->get_relative_pose_tlr().translation();
        Eigen::Vector3f twl = pFrame->get_camera_origin();

        Ow = Rwl * tlr + twl;
    }
    normal_vector = world_position - Ow;
    normal_vector = normal_vector / normal_vector.norm();

    Eigen::Vector3f PC   = world_position - Ow;
    const float     dist = PC.norm();
    const int       level =
        (!pFrame->is_fisheye_stereo || std::cmp_less(idxF, pFrame->keypoints.size())) ? pFrame->keypoints[idxF].octave : pFrame->right_keypoints[idxF - pFrame->keypoints.size()].octave;
    const float levelScaleFactor = pFrame->scale_factors[level];
    const int   nLevels          = pFrame->scale_levels;

    maximum_distance = dist * levelScaleFactor;
    minimum_distance = maximum_distance / pFrame->scale_factors[nLevels - 1];

    pFrame->descriptor(idxF).copyTo(descriptor);
    _color     = pFrame->keypoint_color(idxF);
    _has_color = !(pFrame->is_fisheye_stereo && std::cmp_greater_equal(idxF, pFrame->keypoints.size()) ? pFrame->color_image_right : pFrame->color_image).empty();

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(owner_map->point_creation_mutex);
    id = next_id++;
}

void map_point::set_world_position(const Eigen::Vector3f &Pos)
{
    unique_lock<mutex> lock2(global_mutex);
    unique_lock<mutex> lock(position_mutex);
    world_position = Pos;
}

Eigen::Vector3f map_point::get_world_position()
{
    unique_lock<mutex> lock(position_mutex);
    return world_position;
}

Eigen::Vector3f map_point::get_normal()
{
    unique_lock<mutex> lock(position_mutex);
    return normal_vector;
}

KeyFrame *map_point::get_reference_key_frame()
{
    unique_lock<mutex> lock(features_mutex);
    return reference_key_frame;
}

std::array<unsigned char, 3> map_point::color()
{
    std::scoped_lock lock(features_mutex);
    return _color;
}

void map_point::add_observation(KeyFrame *pKF, int idx)
{
    unique_lock<mutex> lock(features_mutex);
    tuple<int, int>    indexes;

    if (!_has_color && idx >= 0 && static_cast<std::size_t>(idx) < static_cast<std::size_t>(pKF->N))
    {
        const bool  right = is_fisheye_stereo(pKF->cameras) && std::cmp_greater_equal(idx, pKF->mvKeys.size());
        const auto &key   = right ? pKF->mvKeysRight[idx - pKF->mvKeys.size()] : pKF->mvKeys[idx];
        _color            = key.color;
        _has_color        = key.has_color;
    }

    if (observations_by_key_frame.count(pKF))
    {
        indexes = observations_by_key_frame[pKF];
    }
    else
    {
        indexes = tuple<int, int>(-1, -1);
    }

    if (is_fisheye_stereo(pKF->cameras) && std::cmp_greater_equal(idx, pKF->mvKeys.size()))
    {
        get<1>(indexes) = idx;
    }
    else
    {
        get<0>(indexes) = idx;
    }

    observations_by_key_frame[pKF] = indexes;

    if (!is_fisheye_stereo(pKF->cameras) && pKF->mvuRight[idx] >= 0)
    {
        observation_count += 2;
    }
    else
    {
        observation_count++;
    }
}

void map_point::erase_observation(KeyFrame *pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(features_mutex);
        if (observations_by_key_frame.count(pKF))
        {
            tuple<int, int> indexes   = observations_by_key_frame[pKF];
            int             leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (leftIndex != -1)
            {
                if (!is_fisheye_stereo(pKF->cameras) && pKF->mvuRight[leftIndex] >= 0)
                {
                    observation_count -= 2;
                }
                else
                {
                    observation_count--;
                }
            }
            if (rightIndex != -1)
            {
                observation_count--;
            }

            observations_by_key_frame.erase(pKF);

            if (reference_key_frame == pKF)
            {
                reference_key_frame = observations_by_key_frame.begin()->first;
            }

            // If only 2 observations or less, discard point
            if (observation_count <= 2)
            {
                bBad = true;
            }
        }
    }

    if (bBad)
    {
        set_bad_flag();
    }
}

std::map<KeyFrame *, std::tuple<int, int>> map_point::get_observations()
{
    unique_lock<mutex> lock(features_mutex);
    return observations_by_key_frame;
}

int map_point::get_observation_count()
{
    unique_lock<mutex> lock(features_mutex);
    return observation_count;
}

void map_point::set_bad_flag()
{
    std::map<KeyFrame *, tuple<int, int>> obs;
    {
        unique_lock<mutex> lock1(features_mutex);
        unique_lock<mutex> lock2(position_mutex);
        bad = true;
        obs = observations_by_key_frame;
        observations_by_key_frame.clear();
    }
    for (std::map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        KeyFrame *pKF       = mit->first;
        int       leftIndex = get<0>(mit->second), rightIndex = get<1>(mit->second);
        if (leftIndex != -1)
        {
            pKF->EraseMapPointMatch(leftIndex);
        }
        if (rightIndex != -1)
        {
            pKF->EraseMapPointMatch(rightIndex);
        }
    }

    owner_map->erase_map_point(this);
}

map_point *map_point::get_replaced()
{
    unique_lock<mutex> lock1(features_mutex);
    unique_lock<mutex> lock2(position_mutex);
    return replaced;
}

void map_point::replace(map_point *pMP)
{
    if (pMP->id == this->id)
    {
        return;
    }

    int                                   nvisible, nfound;
    std::map<KeyFrame *, tuple<int, int>> obs;
    {
        unique_lock<mutex> lock1(features_mutex);
        unique_lock<mutex> lock2(position_mutex);
        obs = observations_by_key_frame;
        observations_by_key_frame.clear();
        bad      = true;
        nvisible = visible;
        nfound   = found;
        replaced = pMP;
    }

    for (std::map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        // replace measurement in keyframe
        KeyFrame *pKF = mit->first;

        tuple<int, int> indexes   = mit->second;
        int             leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if (!pMP->is_in_key_frame(pKF))
        {
            if (leftIndex != -1)
            {
                pKF->ReplaceMapPointMatch(leftIndex, pMP);
                pMP->add_observation(pKF, leftIndex);
            }
            if (rightIndex != -1)
            {
                pKF->ReplaceMapPointMatch(rightIndex, pMP);
                pMP->add_observation(pKF, rightIndex);
            }
        }
        else
        {
            if (leftIndex != -1)
            {
                pKF->EraseMapPointMatch(leftIndex);
            }
            if (rightIndex != -1)
            {
                pKF->EraseMapPointMatch(rightIndex);
            }
        }
    }
    pMP->increase_found(nfound);
    pMP->increase_visible(nvisible);
    pMP->compute_distinctive_descriptors();

    owner_map->erase_map_point(this);
}

bool map_point::is_bad()
{
    unique_lock<mutex> lock1(features_mutex, std::defer_lock);
    unique_lock<mutex> lock2(position_mutex, std::defer_lock);
    lock(lock1, lock2);

    return bad;
}

void map_point::increase_visible(int n)
{
    unique_lock<mutex> lock(features_mutex);
    visible += n;
}

void map_point::increase_found(int n)
{
    unique_lock<mutex> lock(features_mutex);
    found += n;
}

float map_point::get_found_ratio()
{
    unique_lock<mutex> lock(features_mutex);
    return static_cast<float>(found) / visible;
}

void map_point::compute_distinctive_descriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    std::map<KeyFrame *, tuple<int, int>> observations;

    {
        unique_lock<mutex> lock1(features_mutex);
        if (bad)
        {
            return;
        }
        observations = observations_by_key_frame;
    }

    if (observations.empty())
    {
        return;
    }

    vDescriptors.reserve(observations.size());

    for (std::map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;

        if (!pKF->isBad())
        {
            tuple<int, int> indexes   = mit->second;
            int             leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (leftIndex != -1)
            {
                vDescriptors.push_back(pKF->descriptor(leftIndex));
            }
            if (rightIndex != -1)
            {
                vDescriptors.push_back(pKF->descriptor(rightIndex));
            }
        }
    }

    if (vDescriptors.empty())
    {
        return;
    }

    // Compute distances between them
    const size_t N = vDescriptors.size();

    vector<vector<float>> Distances(N, vector<float>(N));
    for (size_t i = 0; i < N; i++)
    {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++)
        {
            float distij    = descriptor_matcher::descriptor_distance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    float BestMedian = INT_MAX;
    int   BestIdx    = 0;
    for (size_t i = 0; i < N; i++)
    {
        auto vDists = Distances[i];
        sort(vDists.begin(), vDists.end());
        float median = vDists[(N - 1) / 2];

        if (median < BestMedian)
        {
            BestMedian = median;
            BestIdx    = i;
        }
    }

    {
        unique_lock<mutex> lock(features_mutex);
        descriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat map_point::get_descriptor()
{
    unique_lock<mutex> lock(features_mutex);
    return descriptor.clone();
}

tuple<int, int> map_point::get_index_in_key_frame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(features_mutex);
    if (observations_by_key_frame.count(pKF))
    {
        return observations_by_key_frame[pKF];
    }
    else
    {
        return tuple<int, int>(-1, -1);
    }
}

bool map_point::is_in_key_frame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(features_mutex);
    return (observations_by_key_frame.count(pKF));
}

void map_point::update_normal_and_depth()
{
    std::map<KeyFrame *, tuple<int, int>> observations;
    KeyFrame                             *pRefKF;
    Eigen::Vector3f                       Pos;
    {
        unique_lock<mutex> lock1(features_mutex);
        unique_lock<mutex> lock2(position_mutex);
        if (bad)
        {
            return;
        }
        observations = observations_by_key_frame;
        pRefKF       = reference_key_frame;
        Pos          = world_position;
    }

    if (observations.empty())
    {
        return;
    }

    Eigen::Vector3f normal;
    normal.setZero();
    int n = 0;
    for (std::map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;

        tuple<int, int> indexes   = mit->second;
        int             leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if (leftIndex != -1)
        {
            Eigen::Vector3f Owi     = pKF->GetCameraCenter();
            Eigen::Vector3f normali = Pos - Owi;
            normal                  = normal + normali / normali.norm();
            n++;
        }
        if (rightIndex != -1)
        {
            Eigen::Vector3f Owi     = pKF->GetRightCameraCenter();
            Eigen::Vector3f normali = Pos - Owi;
            normal                  = normal + normali / normali.norm();
            n++;
        }
    }

    Eigen::Vector3f PC   = Pos - pRefKF->GetCameraCenter();
    const float     dist = PC.norm();

    tuple<int, int> indexes   = observations[pRefKF];
    int             leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    int             level;
    if (!is_fisheye_stereo(pRefKF->cameras))
    {
        level = pRefKF->mvKeys[leftIndex].octave;
    }
    else if (leftIndex != -1)
    {
        level = pRefKF->mvKeys[leftIndex].octave;
    }
    else
    {
        level = pRefKF->mvKeysRight[rightIndex - pRefKF->mvKeys.size()].octave;
    }

    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int   nLevels          = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(position_mutex);
        maximum_distance = dist * levelScaleFactor;
        minimum_distance = maximum_distance / pRefKF->mvScaleFactors[nLevels - 1];
        normal_vector    = normal / n;
    }
}

void map_point::set_normal_vector(const Eigen::Vector3f &normal)
{
    unique_lock<mutex> lock3(position_mutex);
    normal_vector = normal;
}

float map_point::get_min_distance_invariance()
{
    unique_lock<mutex> lock(position_mutex);
    return 0.8f * minimum_distance;
}

float map_point::get_max_distance_invariance()
{
    unique_lock<mutex> lock(position_mutex);
    return 1.2f * maximum_distance;
}

int map_point::predict_scale(const float &currentDist, KeyFrame *pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(position_mutex);
        ratio = maximum_distance / currentDist;
    }

    int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
    if (nScale < 0)
    {
        nScale = 0;
    }
    else if (nScale >= pKF->mnScaleLevels)
    {
        nScale = pKF->mnScaleLevels - 1;
    }

    return nScale;
}

int map_point::predict_scale(const float &currentDist, frame *pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(position_mutex);
        ratio = maximum_distance / currentDist;
    }

    int nScale = ceil(log(ratio) / pF->log_scale_factor);
    if (nScale < 0)
    {
        nScale = 0;
    }
    else if (nScale >= pF->scale_levels)
    {
        nScale = pF->scale_levels - 1;
    }

    return nScale;
}

void map_point::print_observations()
{
    spdlog::debug("Observations for map point {}", id);
    for (std::map<KeyFrame *, tuple<int, int>>::iterator mit = observations_by_key_frame.begin(), mend = observations_by_key_frame.end(); mit != mend; mit++)
    {
        KeyFrame            *pKFi      = mit->first;
        tuple<int, int>      indexes   = mit->second;
        [[maybe_unused]] int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        spdlog::debug("Observed in keyframe {} in map {}", pKFi->mnId, pKFi->GetMap()->get_id());
    }
}

map *map_point::get_map()
{
    unique_lock<mutex> lock(map_mutex);
    return owner_map;
}

void map_point::update_map(map *pMap)
{
    unique_lock<mutex> lock(map_mutex);
    owner_map = pMap;
}

} // namespace ORB_SLAM3
