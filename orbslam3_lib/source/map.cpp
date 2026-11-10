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

#include "map.h"

#include <mutex>
#include <utility>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

long unsigned int map::next_id = 0;

map::map()
    : first_region_key_frame(nullptr), failed(false), imu_initialized(false), map_change_index(0), notified_map_change_index(0), max_key_frame_id(0), big_change_index(0), in_use(false),
      bad(false), inertial(false), imu_ba1(false), imu_ba2(false)
{
    id = next_id++;
}

map::map(int initKFid)
    : first_region_key_frame(nullptr), failed(false), imu_initialized(false), map_change_index(0), notified_map_change_index(0), initial_key_frame_id(initKFid), max_key_frame_id(initKFid),
      big_change_index(0), in_use(false), bad(false), inertial(false), imu_ba1(false), imu_ba2(false)
{
    id = next_id++;
}

map::~map()
{
    // TODO: erase all points from memory
    map_points.clear();

    // TODO: erase all keyframes from memory
    key_frames.clear();

    reference_map_points.clear();
    key_frame_origins.clear();
}

void map::add_key_frame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(map_mutex);
    if (key_frames.empty())
    {
        spdlog::debug("First keyframe: {}; map initial keyframe: {}", pKF->mnId, initial_key_frame_id);
        initial_key_frame_id = pKF->mnId;
        initial_key_frame    = pKF;
        lowest_id_key_frame  = pKF;
    }
    key_frames.insert(pKF);
    if (pKF->mnId > max_key_frame_id)
    {
        max_key_frame_id = pKF->mnId;
    }
    if (pKF->mnId < lowest_id_key_frame->mnId)
    {
        lowest_id_key_frame = pKF;
    }
}

void map::add_map_point(MapPoint *pMP)
{
    unique_lock<mutex> lock(map_mutex);
    map_points.insert(pMP);
}

void map::set_imu_initialized()
{
    unique_lock<mutex> lock(map_mutex);
    imu_initialized = true;
}

bool map::is_imu_initialized()
{
    unique_lock<mutex> lock(map_mutex);
    return imu_initialized;
}

void map::erase_map_point(MapPoint *pMP)
{
    unique_lock<mutex> lock(map_mutex);
    map_points.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void map::erase_key_frame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(map_mutex);
    key_frames.erase(pKF);
    if (key_frames.size() > 0)
    {
        if (pKF->mnId == lowest_id_key_frame->mnId)
        {
            vector<KeyFrame *> vpKFs = vector<KeyFrame *>(key_frames.begin(), key_frames.end());
            sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
            lowest_id_key_frame = vpKFs[0];
        }
    }
    else
    {
        lowest_id_key_frame = 0;
    }

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void map::set_reference_map_points(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(map_mutex);
    reference_map_points = vpMPs;
}

void map::inform_new_big_change()
{
    unique_lock<mutex> lock(map_mutex);
    big_change_index++;
}

int map::get_last_big_change_index()
{
    unique_lock<mutex> lock(map_mutex);
    return big_change_index;
}

vector<KeyFrame *> map::get_all_key_frames()
{
    unique_lock<mutex> lock(map_mutex);
    return vector<KeyFrame *>(key_frames.begin(), key_frames.end());
}

vector<MapPoint *> map::get_all_map_points()
{
    unique_lock<mutex> lock(map_mutex);
    return vector<MapPoint *>(map_points.begin(), map_points.end());
}

long unsigned int map::map_point_count()
{
    unique_lock<mutex> lock(map_mutex);
    return map_points.size();
}

long unsigned int map::key_frame_count()
{
    unique_lock<mutex> lock(map_mutex);
    return key_frames.size();
}

vector<MapPoint *> map::get_reference_map_points()
{
    unique_lock<mutex> lock(map_mutex);
    return reference_map_points;
}

long unsigned int map::get_id() { return id; }
long unsigned int map::get_initial_key_frame_id()
{
    unique_lock<mutex> lock(map_mutex);
    return initial_key_frame_id;
}

void map::set_initial_key_frame_id(long unsigned int initKFif)
{
    unique_lock<mutex> lock(map_mutex);
    initial_key_frame_id = initKFif;
}

long unsigned int map::get_max_key_frame_id()
{
    unique_lock<mutex> lock(map_mutex);
    return max_key_frame_id;
}

KeyFrame *map::get_origin_key_frame() { return initial_key_frame; }

void map::set_current_map() { in_use = true; }

void map::set_stored_map() { in_use = false; }

void map::clear()
{
    for (set<KeyFrame *>::iterator sit = key_frames.begin(), send = key_frames.end(); sit != send; sit++)
    {
        KeyFrame *pKF = *sit;
        pKF->UpdateMap(static_cast<map *>(NULL));
    }

    map_points.clear();
    key_frames.clear();
    max_key_frame_id = initial_key_frame_id;
    imu_initialized  = false;
    reference_map_points.clear();
    key_frame_origins.clear();
    imu_ba1 = false;
    imu_ba2 = false;
}

bool map::is_in_use() { return in_use; }

void map::set_bad() { bad = true; }

bool map::is_bad() { return bad; }

void map::apply_scaled_rotation(const Sophus::SE3f &T, const float s, const bool bScaledVel)
{
    unique_lock<mutex> lock(map_mutex);

    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    Sophus::SE3f    Tyw = T;
    Eigen::Matrix3f Ryw = Tyw.rotationMatrix();
    Eigen::Vector3f tyw = Tyw.translation();

    for (set<KeyFrame *>::iterator sit = key_frames.begin(); sit != key_frames.end(); sit++)
    {
        KeyFrame    *pKF   = *sit;
        Sophus::SE3f Twc   = pKF->GetPoseInverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc   = Tyw * Twc;
        Sophus::SE3f Tcy   = Tyc.inverse();
        pKF->SetPose(Tcy);
        Eigen::Vector3f Vw = pKF->GetVelocity();
        if (!bScaledVel)
        {
            pKF->SetVelocity(Ryw * Vw);
        }
        else
        {
            pKF->SetVelocity(Ryw * Vw * s);
        }
    }
    for (set<MapPoint *>::iterator sit = map_points.begin(); sit != map_points.end(); sit++)
    {
        MapPoint *pMP = *sit;
        pMP->SetWorldPos(s * Ryw * pMP->GetWorldPos() + tyw);
        pMP->UpdateNormalAndDepth();
    }
    map_change_index++;
}

void map::set_inertial_sensor()
{
    unique_lock<mutex> lock(map_mutex);
    inertial = true;
}

bool map::is_inertial()
{
    unique_lock<mutex> lock(map_mutex);
    return inertial;
}

void map::set_inertial_ba1()
{
    unique_lock<mutex> lock(map_mutex);
    imu_ba1 = true;
}

void map::set_inertial_ba2()
{
    unique_lock<mutex> lock(map_mutex);
    imu_ba2 = true;
}

bool map::get_inertial_ba1()
{
    unique_lock<mutex> lock(map_mutex);
    return imu_ba1;
}

bool map::get_inertial_ba2()
{
    unique_lock<mutex> lock(map_mutex);
    return imu_ba2;
}

void map::change_id(long unsigned int nId) { id = nId; }

unsigned int map::get_lowest_key_frame_id()
{
    unique_lock<mutex> lock(map_mutex);
    if (lowest_id_key_frame)
    {
        return lowest_id_key_frame->mnId;
    }
    return 0;
}

int map::get_map_change_index()
{
    unique_lock<mutex> lock(map_mutex);
    return map_change_index;
}

void map::increase_change_index()
{
    unique_lock<mutex> lock(map_mutex);
    map_change_index++;
}

int map::get_last_map_change()
{
    unique_lock<mutex> lock(map_mutex);
    return notified_map_change_index;
}

void map::set_last_map_change(int currentChangeId)
{
    unique_lock<mutex> lock(map_mutex);
    notified_map_change_index = currentChangeId;
}

} // namespace ORB_SLAM3
