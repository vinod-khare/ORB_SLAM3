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

#include "key_frame.h"

#include "converter.h"
#include "imu_types.h"

#include <mutex>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

namespace
{
std::vector<keypoint> keypoints_with_color(frame &frame, const std::vector<keypoint> &keys, int index_offset)
{
    auto colored_keys = keys;
    for (std::size_t index = 0; index < colored_keys.size(); ++index)
    {
        const int frame_index         = index_offset + static_cast<int>(index);
        colored_keys[index].color     = frame.keypoint_color(frame_index);
        const auto &image             = frame.is_fisheye_stereo && frame_index >= static_cast<int>(frame.keypoints.size()) ? frame.color_image_right : frame.color_image;
        colored_keys[index].has_color = !image.empty();
    }
    return colored_keys;
}
} // namespace

long unsigned int key_frame::next_id = 0;

key_frame::key_frame(frame &F, map *pMap, key_frame_database *pKFDB, const imu::calib &imu_calibration)
    : has_imu(pMap->is_imu_initialized()), frame_id(F.id), timestamp(F.timestamp), grid_columns(FRAME_GRID_COLS), grid_rows(FRAME_GRID_ROWS),
      grid_element_width_inverse(F.grid_element_width_inverse), grid_element_height_inverse(F.grid_element_height_inverse), track_reference_for_frame(0), fuse_target_key_frame_id(0),
      local_ba_key_frame_id(0), fixed_ba_key_frame_id(0), number_of_optimizations(0), loop_query_id(0), loop_common_words(0), relocalization_query_id(0), relocalization_words(0),
      place_recognition_query_id(0), place_recognition_words(0), place_recognition_score(0), current_place_recognition(false), global_ba_key_frame_id(0), merge_corrected_key_frame_id(0),
      local_ba_merge_key_frame_id(0), feature_count(F.feature_count), is_fisheye_stereo(F.is_fisheye_stereo), keypoints(keypoints_with_color(F, F.keypoints, 0)),
      right_coordinates(F.right_coordinates), depths(F.depths), bow_vector(F.bow_vector), feature_vector(F.feature_vector), scale_levels(F.scale_levels), scale_factor(F.scale_factor),
      log_scale_factor(F.log_scale_factor), scale_factors(F.scale_factors), level_sigma_squared(F.level_sigma_squared), inverse_level_sigma_squared(F.inverse_level_sigma_squared),
      min_x(F.min_x), min_y(F.min_y), max_x(F.max_x), max_y(F.max_y), previous_key_frame(nullptr), next_key_frame(nullptr), imu_preintegrated(F.imu_preintegrated),
      imu_calibration(imu_calibration), name_file(F.name_file), dataset(F.dataset), has_velocity(false), map_points(F.map_points), database(pKFDB),
      place_vocabulary_ptr(F.place_vocabulary_ptr), first_connection(true), parent_key_frame(nullptr), not_erasable(false), to_be_erased(false), bad(false), owner_map(pMap),
      left_to_right_matches(F.left_to_right_matches), right_to_left_matches(F.right_to_left_matches),
      right_keypoints(keypoints_with_color(F, F.right_keypoints, static_cast<int>(F.keypoints.size())))
{
    id = next_id++;
    grid.resize(grid_columns);
    if (is_fisheye_stereo)
    {
        right_grid.resize(grid_columns);
    }
    for (int i = 0; i < grid_columns; i++)
    {
        grid[i].resize(grid_rows);
        if (is_fisheye_stereo)
        {
            right_grid[i].resize(grid_rows);
        }
        for (int j = 0; j < grid_rows; j++)
        {
            grid[i][j] = F.grid[i][j];
            if (is_fisheye_stereo)
            {
                right_grid[i][j] = F.right_grid[i][j];
            }
        }
    }

    if (!F.has_velocity())
    {
        velocity_world.setZero();
        has_velocity = false;
    }
    else
    {
        velocity_world = F.get_velocity();
        has_velocity   = true;
    }

    imu_bias = F.imu_bias;
    set_pose(F.get_pose());

    origin_map_id = pMap->get_id();
}

void key_frame::compute_bow()
{
    if (bow_vector.empty() || feature_vector.empty())
    {
        auto       vCurrentDesc      = descriptor_vector(keypoints);
        const auto right_descriptors = descriptor_vector(right_keypoints);
        vCurrentDesc.insert(vCurrentDesc.end(), right_descriptors.begin(), right_descriptors.end());
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        place_vocabulary_ptr->transform(vCurrentDesc, bow_vector, feature_vector, 4);
    }
}

void key_frame::set_pose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(pose_mutex);

    camera_pose     = Tcw;
    camera_rotation = camera_pose.rotationMatrix();
    world_pose      = camera_pose.inverse();
    world_rotation  = world_pose.rotationMatrix();

    if (imu_calibration.is_set) // TODO Use a flag instead of the OpenCV matrix
    {
        imu_position_world = world_rotation * imu_calibration.camera_to_body.translation() + world_pose.translation();
    }
}

void key_frame::set_velocity(const Eigen::Vector3f &Vw)
{
    unique_lock<mutex> lock(pose_mutex);
    velocity_world = Vw;
    has_velocity   = true;
}

Sophus::SE3f key_frame::get_pose()
{
    unique_lock<mutex> lock(pose_mutex);
    return camera_pose;
}

Sophus::SE3f key_frame::get_pose_inverse()
{
    unique_lock<mutex> lock(pose_mutex);
    return world_pose;
}

Eigen::Vector3f key_frame::get_camera_center()
{
    unique_lock<mutex> lock(pose_mutex);
    return world_pose.translation();
}

Eigen::Vector3f key_frame::get_imu_position()
{
    unique_lock<mutex> lock(pose_mutex);
    return imu_position_world;
}

Eigen::Matrix3f key_frame::get_imu_rotation()
{
    unique_lock<mutex> lock(pose_mutex);
    return (world_pose * imu_calibration.camera_to_body).rotationMatrix();
}

Sophus::SE3f key_frame::get_imu_pose()
{
    unique_lock<mutex> lock(pose_mutex);
    return world_pose * imu_calibration.camera_to_body;
}

Eigen::Matrix3f key_frame::get_rotation()
{
    unique_lock<mutex> lock(pose_mutex);
    return camera_rotation;
}

Eigen::Vector3f key_frame::get_translation()
{
    unique_lock<mutex> lock(pose_mutex);
    return camera_pose.translation();
}

Eigen::Vector3f key_frame::get_velocity()
{
    unique_lock<mutex> lock(pose_mutex);
    return velocity_world;
}

bool key_frame::is_velocity_set()
{
    unique_lock<mutex> lock(pose_mutex);
    return has_velocity;
}

void key_frame::add_connection(key_frame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(connections_mutex);
        if (!connected_key_frame_weights.count(pKF))
        {
            connected_key_frame_weights[pKF] = weight;
        }
        else if (connected_key_frame_weights[pKF] != weight)
        {
            connected_key_frame_weights[pKF] = weight;
        }
        else
        {
            return;
        }
    }

    update_best_covisibles();
}

void key_frame::update_best_covisibles()
{
    unique_lock<mutex>             lock(connections_mutex);
    vector<pair<int, key_frame *>> vPairs;
    vPairs.reserve(connected_key_frame_weights.size());
    for (std::map<key_frame *, int>::iterator mit = connected_key_frame_weights.begin(), mend = connected_key_frame_weights.end(); mit != mend; mit++)
    {
        vPairs.push_back(make_pair(mit->second, mit->first));
    }

    sort(vPairs.begin(), vPairs.end());
    list<key_frame *> lKFs;
    list<int>         lWs;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        if (!vPairs[i].second->is_bad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    ordered_connected_key_frames = vector<key_frame *>(lKFs.begin(), lKFs.end());
    ordered_weights              = vector<int>(lWs.begin(), lWs.end());
}

set<key_frame *> key_frame::get_connected_key_frames()
{
    unique_lock<mutex> lock(connections_mutex);
    set<key_frame *>   s;
    for (std::map<key_frame *, int>::iterator mit = connected_key_frame_weights.begin(); mit != connected_key_frame_weights.end(); mit++)
    {
        s.insert(mit->first);
    }
    return s;
}

vector<key_frame *> key_frame::get_vector_covisible_key_frames()
{
    unique_lock<mutex> lock(connections_mutex);
    return ordered_connected_key_frames;
}

vector<key_frame *> key_frame::get_best_covisibility_key_frames(const int &number_of_key_frames)
{
    unique_lock<mutex> lock(connections_mutex);
    if ((int)ordered_connected_key_frames.size() < number_of_key_frames)
    {
        return ordered_connected_key_frames;
    }
    else
    {
        return vector<key_frame *>(ordered_connected_key_frames.begin(), ordered_connected_key_frames.begin() + number_of_key_frames);
    }
}

vector<key_frame *> key_frame::get_covisibles_by_weight(const int &w)
{
    unique_lock<mutex> lock(connections_mutex);

    if (ordered_connected_key_frames.empty())
    {
        return vector<key_frame *>();
    }

    vector<int>::iterator it = upper_bound(ordered_weights.begin(), ordered_weights.end(), w, key_frame::compare_weight);

    if (it == ordered_weights.end() && ordered_weights.back() < w)
    {
        return vector<key_frame *>();
    }
    else
    {
        int n = it - ordered_weights.begin();
        return vector<key_frame *>(ordered_connected_key_frames.begin(), ordered_connected_key_frames.begin() + n);
    }
}

int key_frame::get_weight(key_frame *pKF)
{
    unique_lock<mutex> lock(connections_mutex);
    if (connected_key_frame_weights.count(pKF))
    {
        return connected_key_frame_weights[pKF];
    }
    else
    {
        return 0;
    }
}

int key_frame::get_map_point_count()
{
    unique_lock<mutex> lock(features_mutex);
    int                numberMPs = 0;
    for (size_t i = 0, iend = map_points.size(); i < iend; i++)
    {
        if (!map_points[i])
        {
            continue;
        }
        numberMPs++;
    }
    return numberMPs;
}

void key_frame::add_map_point(map_point *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(features_mutex);
    map_points[idx] = pMP;
}

void key_frame::erase_map_point_match(const int &idx)
{
    unique_lock<mutex> lock(features_mutex);
    map_points[idx] = static_cast<map_point *>(NULL);
}

void key_frame::erase_map_point_match(map_point *pMP)
{
    const auto indexes   = pMP->get_index_in_key_frame(this);
    const int  leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    if (leftIndex != -1)
    {
        map_points[leftIndex] = static_cast<map_point *>(NULL);
    }
    if (rightIndex != -1)
    {
        map_points[rightIndex] = static_cast<map_point *>(NULL);
    }
}

void key_frame::replace_map_point_match(const int &idx, map_point *pMP) { map_points[idx] = pMP; }

set<map_point *> key_frame::get_map_points()
{
    unique_lock<mutex> lock(features_mutex);
    set<map_point *>   s;
    for (size_t i = 0, iend = map_points.size(); i < iend; i++)
    {
        if (!map_points[i])
        {
            continue;
        }
        map_point *pMP = map_points[i];
        if (!pMP->is_bad())
        {
            s.insert(pMP);
        }
    }
    return s;
}

int key_frame::tracked_map_points(const int &minObs)
{
    unique_lock<mutex> lock(features_mutex);

    int        nPoints   = 0;
    const bool bCheckObs = minObs > 0;
    for (int i = 0; i < feature_count; i++)
    {
        map_point *pMP = map_points[i];
        // TODO: reduce nesting
        if (pMP)
        {
            if (!pMP->is_bad())
            {
                if (bCheckObs)
                {
                    if (map_points[i]->get_observation_count() >= minObs)
                    {
                        nPoints++;
                    }
                }
                else
                {
                    nPoints++;
                }
            }
        }
    }

    return nPoints;
}

vector<map_point *> key_frame::get_map_point_matches()
{
    unique_lock<mutex> lock(features_mutex);
    return map_points;
}

map_point *key_frame::get_map_point(const size_t &idx)
{
    unique_lock<mutex> lock(features_mutex);
    return map_points[idx];
}

void key_frame::update_connections(bool upParent)
{
    std::map<key_frame *, int> KFcounter;

    vector<map_point *> vpMP;

    {
        unique_lock<mutex> lockMPs(features_mutex);
        vpMP = map_points;
    }

    // For all map points in keyframe check in which other keyframes are they seen
    // Increase counter for those keyframes
    for (vector<map_point *>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
    {
        map_point *pMP = *vit;

        if (!pMP)
        {
            continue;
        }

        if (pMP->is_bad())
        {
            continue;
        }

        std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();

        for (std::map<key_frame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            if (mit->first->id == id || mit->first->is_bad() || mit->first->get_map() != owner_map)
            {
                continue;
            }
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if (KFcounter.empty())
    {
        return;
    }

    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum counter
    int        nmax   = 0;
    key_frame *pKFmax = NULL;
    int        th     = 15;

    vector<pair<int, key_frame *>> vPairs;
    vPairs.reserve(KFcounter.size());
    if (!upParent)
    {
        spdlog::debug("Updating connections for keyframe {}", id);
    }

    // TODO: use modern iterators
    for (std::map<key_frame *, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
    {
        if (!upParent)
        {
            spdlog::trace("Connected keyframe {} has {} matches", mit->first->id, mit->second);
        }
        if (mit->second > nmax)
        {
            nmax   = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th)
        {
            vPairs.push_back(make_pair(mit->second, mit->first));
            (mit->first)->add_connection(this, mit->second);
        }
    }

    if (vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->add_connection(this, nmax);
    }

    sort(vPairs.begin(), vPairs.end());
    list<key_frame *> lKFs;
    list<int>         lWs;
    for (size_t i = 0; i < vPairs.size(); i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(connections_mutex);

        connected_key_frame_weights  = KFcounter;
        ordered_connected_key_frames = vector<key_frame *>(lKFs.begin(), lKFs.end());
        ordered_weights              = vector<int>(lWs.begin(), lWs.end());

        if (first_connection && id != owner_map->get_initial_key_frame_id())
        {
            parent_key_frame = ordered_connected_key_frames.front();
            parent_key_frame->add_child(this);
            first_connection = false;
        }
    }
}

void key_frame::add_child(key_frame *pKF)
{
    unique_lock<mutex> lockCon(connections_mutex);
    child_key_frames.insert(pKF);
}

void key_frame::erase_child(key_frame *pKF)
{
    unique_lock<mutex> lockCon(connections_mutex);
    child_key_frames.erase(pKF);
}

void key_frame::change_parent(key_frame *pKF)
{
    unique_lock<mutex> lockCon(connections_mutex);
    if (pKF == this)
    {
        spdlog::error("Cannot set a keyframe as its own parent");
        throw std::invalid_argument("The parent and child can not be the same");
    }

    parent_key_frame = pKF;
    pKF->add_child(this);
}

set<key_frame *> key_frame::get_children()
{
    unique_lock<mutex> lockCon(connections_mutex);
    return child_key_frames;
}

key_frame *key_frame::get_parent()
{
    unique_lock<mutex> lockCon(connections_mutex);
    return parent_key_frame;
}

bool key_frame::has_child(key_frame *pKF)
{
    unique_lock<mutex> lockCon(connections_mutex);
    return child_key_frames.count(pKF);
}

void key_frame::set_first_connection(bool bFirst)
{
    unique_lock<mutex> lockCon(connections_mutex);
    first_connection = bFirst;
}

void key_frame::add_loop_edge(key_frame *pKF)
{
    unique_lock<mutex> lockCon(connections_mutex);
    not_erasable = true;
    loop_edges.insert(pKF);
}

set<key_frame *> key_frame::get_loop_edges()
{
    unique_lock<mutex> lockCon(connections_mutex);
    return loop_edges;
}

void key_frame::add_merge_edge(key_frame *pKF)
{
    unique_lock<mutex> lockCon(connections_mutex);
    not_erasable = true;
    merge_edges.insert(pKF);
}

set<key_frame *> key_frame::get_merge_edges()
{
    unique_lock<mutex> lockCon(connections_mutex);
    return merge_edges;
}

void key_frame::set_not_erase()
{
    unique_lock<mutex> lock(connections_mutex);
    not_erasable = true;
}

void key_frame::set_erase()
{
    {
        unique_lock<mutex> lock(connections_mutex);
        if (loop_edges.empty())
        {
            not_erasable = false;
        }
    }

    if (to_be_erased)
    {
        set_bad_flag();
    }
}

void key_frame::set_bad_flag()
{
    {
        unique_lock<mutex> lock(connections_mutex);
        if (id == owner_map->get_initial_key_frame_id())
        {
            return;
        }
        else if (not_erasable)
        {
            to_be_erased = true;
            return;
        }
    }

    for (std::map<key_frame *, int>::iterator mit = connected_key_frame_weights.begin(), mend = connected_key_frame_weights.end(); mit != mend; mit++)
    {
        mit->first->erase_connection(this);
    }

    for (size_t i = 0; i < map_points.size(); i++)
    {
        if (map_points[i])
        {
            map_points[i]->erase_observation(this);
        }
    }

    {
        unique_lock<mutex> lock(connections_mutex);
        unique_lock<mutex> lock1(features_mutex);

        connected_key_frame_weights.clear();
        ordered_connected_key_frames.clear();

        // Update Spanning Tree
        set<key_frame *> sParentCandidates;
        if (parent_key_frame)
        {
            sParentCandidates.insert(parent_key_frame);
        }

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while (!child_key_frames.empty())
        {
            bool bContinue = false;

            int        max = -1;
            key_frame *pC  = nullptr;
            key_frame *pP  = nullptr;

            for (set<key_frame *>::iterator sit = child_key_frames.begin(), send = child_key_frames.end(); sit != send; sit++)
            {
                key_frame *pKF = *sit;
                if (pKF->is_bad())
                {
                    continue;
                }

                // Check if a parent candidate is connected to the keyframe
                vector<key_frame *> vpConnected = pKF->get_vector_covisible_key_frames();
                for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
                {
                    for (set<key_frame *>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
                    {
                        if (vpConnected[i]->id == (*spcit)->id)
                        {
                            int w = pKF->get_weight(vpConnected[i]);
                            if (w > max)
                            {
                                pC        = pKF;
                                pP        = vpConnected[i];
                                max       = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if (bContinue)
            {
                pC->change_parent(pP);
                sParentCandidates.insert(pC);
                child_key_frames.erase(pC);
            }
            else
            {
                break;
            }
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if (!child_key_frames.empty())
        {
            for (set<key_frame *>::iterator sit = child_key_frames.begin(); sit != child_key_frames.end(); sit++)
            {
                (*sit)->change_parent(parent_key_frame);
            }
        }

        if (parent_key_frame)
        {
            parent_key_frame->erase_child(this);
            pose_relative_to_parent = camera_pose * parent_key_frame->get_pose_inverse();
        }
        bad = true;
    }

    owner_map->erase_key_frame(this);
    database->erase(this);
}

bool key_frame::is_bad()
{
    unique_lock<mutex> lock(connections_mutex);
    return bad;
}

void key_frame::erase_connection(key_frame *pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(connections_mutex);
        if (connected_key_frame_weights.count(pKF))
        {
            connected_key_frame_weights.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
    {
        update_best_covisibles();
    }
}

vector<size_t> key_frame::get_features_in_area(const float &x, const float &y, const float &r, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(feature_count);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0, (int)floor((x - min_x - factorX) * grid_element_width_inverse));
    if (nMinCellX >= grid_columns)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)grid_columns - 1, (int)ceil((x - min_x + factorX) * grid_element_width_inverse));
    if (nMaxCellX < 0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0, (int)floor((y - min_y - factorY) * grid_element_height_inverse));
    if (nMinCellY >= grid_rows)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)grid_rows - 1, (int)ceil((y - min_y + factorY) * grid_element_height_inverse));
    if (nMaxCellY < 0)
    {
        return vIndices;
    }

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? grid[ix][iy] : right_grid[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const auto &kpUn  = (!is_fisheye_stereo || !bRight) ? keypoints[vCell[j]] : right_keypoints[vCell[j]];
                const float distx = kpUn.pt_undistorted.x - x;
                const float disty = kpUn.pt_undistorted.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                {
                    vIndices.push_back(vCell[j]);
                }
            }
        }
    }

    return vIndices;
}

bool key_frame::is_in_image(const float &x, const float &y) const { return (x >= min_x && x < max_x && y >= min_y && y < max_y); }

bool key_frame::unproject_stereo(int i, Eigen::Vector3f &x3D, geometric_camera &camera)
{
    const float z = depths[i];
    if (z > 0)
    {
        const float     u = keypoints[i].pt.x;
        const float     v = keypoints[i].pt.y;
        const float     x = (u - camera.get_parameter(2)) * z / camera.get_parameter(0);
        const float     y = (v - camera.get_parameter(3)) * z / camera.get_parameter(1);
        Eigen::Vector3f x3Dc(x, y, z);

        unique_lock<mutex> lock(pose_mutex);
        x3D = world_rotation * x3Dc + world_pose.translation();
        return true;
    }
    else
    {
        return false;
    }
}

float key_frame::compute_scene_median_depth(const int q)
{
    if (feature_count == 0)
    {
        return -1.0;
    }

    vector<map_point *> vpMapPoints;
    Eigen::Matrix3f     Rcw;
    Eigen::Vector3f     tcw;
    {
        unique_lock<mutex> lock(features_mutex);
        unique_lock<mutex> lock2(pose_mutex);
        vpMapPoints = map_points;
        tcw         = camera_pose.translation();
        Rcw         = camera_rotation;
    }

    vector<float> vDepths;
    vDepths.reserve(feature_count);
    Eigen::Matrix<float, 1, 3> Rcw2 = Rcw.row(2);
    float                      zcw  = tcw(2);
    for (int i = 0; i < feature_count; i++)
    {
        if (map_points[i])
        {
            map_point      *pMP  = map_points[i];
            Eigen::Vector3f x3Dw = pMP->get_world_position();
            float           z    = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(), vDepths.end());

    return vDepths[(vDepths.size() - 1) / q];
}

void key_frame::set_new_bias(const imu::bias &b)
{
    unique_lock<mutex> lock(pose_mutex);
    imu_bias = b;
    if (imu_preintegrated)
    {
        imu_preintegrated->set_new_bias(b);
    }
}

Eigen::Vector3f key_frame::get_gyro_bias()
{
    unique_lock<mutex> lock(pose_mutex);
    return Eigen::Vector3f(imu_bias.bwx, imu_bias.bwy, imu_bias.bwz);
}

Eigen::Vector3f key_frame::get_accelerometer_bias()
{
    unique_lock<mutex> lock(pose_mutex);
    return Eigen::Vector3f(imu_bias.bax, imu_bias.bay, imu_bias.baz);
}

imu::bias key_frame::get_imu_bias()
{
    unique_lock<mutex> lock(pose_mutex);
    return imu_bias;
}

map *key_frame::get_map()
{
    unique_lock<mutex> lock(map_mutex);
    return owner_map;
}

void key_frame::update_map(map *pMap)
{
    unique_lock<mutex> lock(map_mutex);
    owner_map = pMap;
}

bool key_frame::project_point_distort(map_point *pMP, cv::Point2f &kp, float &u, float &v, geometric_camera &camera)
{
    const float fx                      = camera.get_parameter(0);
    const float fy                      = camera.get_parameter(1);
    const float cx                      = camera.get_parameter(2);
    const float cy                      = camera.get_parameter(3);
    const auto &distortion_coefficients = camera.distortion_coefficients();

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->get_world_position();

    // 3D in camera coordinates
    Eigen::Vector3f Pc  = camera_rotation * P + camera_pose.translation();
    float          &PcX = Pc(0);
    float          &PcY = Pc(1);
    float          &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
        spdlog::debug("Negative depth: {}", PcZ);
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f / PcZ;
    u          = fx * PcX * invz + cx;
    v          = fy * PcY * invz + cy;

    if (u < min_x || u > max_x)
    {
        return false;
    }
    if (v < min_y || v > max_y)
    {
        return false;
    }

    float x  = (u - cx) / fx;
    float y  = (v - cy) / fy;
    float r2 = x * x + y * y;
    float k1 = distortion_coefficients.at<float>(0);
    float k2 = distortion_coefficients.at<float>(1);
    float p1 = distortion_coefficients.at<float>(2);
    float p2 = distortion_coefficients.at<float>(3);
    float k3 = 0;
    if (distortion_coefficients.total() == 5)
    {
        k3 = distortion_coefficients.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    float u_distort = x_distort * fx + cx;
    float v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool key_frame::project_point_undistort(map_point *pMP, cv::Point2f &kp, float &u, float &v, geometric_camera &camera)
{
    const float fx = camera.get_parameter(0);
    const float fy = camera.get_parameter(1);
    const float cx = camera.get_parameter(2);
    const float cy = camera.get_parameter(3);

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->get_world_position();

    // 3D in camera coordinates
    Eigen::Vector3f Pc  = camera_rotation * P + camera_pose.translation();
    float          &PcX = Pc(0);
    float          &PcY = Pc(1);
    float          &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
        spdlog::debug("Negative depth: {}", PcZ);
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    u                = fx * PcX * invz + cx;
    v                = fy * PcY * invz + cy;

    if (u < min_x || u > max_x)
    {
        return false;
    }
    if (v < min_y || v > max_y)
    {
        return false;
    }

    kp = cv::Point2f(u, v);

    return true;
}

Sophus::SE3f key_frame::get_relative_pose_trl(const geometric_camera &camera)
{
    unique_lock<mutex> lock(pose_mutex);
    return camera.left_to_right_pose().inverse();
}

Sophus::SE3f key_frame::get_relative_pose_tlr(const geometric_camera &camera)
{
    unique_lock<mutex> lock(pose_mutex);
    return camera.left_to_right_pose();
}

Sophus::SE3<float> key_frame::get_right_pose(const geometric_camera &camera)
{
    unique_lock<mutex> lock(pose_mutex);

    return camera.left_to_right_pose().inverse() * camera_pose;
}

Sophus::SE3<float> key_frame::get_right_pose_inverse(const geometric_camera &camera)
{
    unique_lock<mutex> lock(pose_mutex);

    return world_pose * camera.left_to_right_pose();
}

Eigen::Vector3f key_frame::get_right_camera_center(const geometric_camera &camera)
{
    unique_lock<mutex> lock(pose_mutex);

    return (world_pose * camera.left_to_right_pose()).translation();
}

Eigen::Matrix<float, 3, 3> key_frame::get_right_rotation(const geometric_camera &camera)
{
    unique_lock<mutex> lock(pose_mutex);

    return (camera.left_to_right_pose().inverse().so3() * camera_pose.so3()).matrix();
}

Eigen::Vector3f key_frame::get_right_translation(const geometric_camera &camera)
{
    unique_lock<mutex> lock(pose_mutex);
    return (camera.left_to_right_pose().inverse() * camera_pose).translation();
}

void key_frame::set_key_frame_database(key_frame_database *pKFDB) { database = pKFDB; }

} // namespace ORB_SLAM3
