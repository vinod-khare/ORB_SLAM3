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

#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"

#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace ORB_SLAM3
{
/** Owns maps and synchronizes access to the active map. */
class atlas
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    atlas() = default;
    atlas(int initial_key_frame_id);
    ~atlas();

    void create_new_map();
    void change_map(Map *map);

    unsigned long int get_last_init_kf_id();

    void add_key_frame(KeyFrame *key_frame);
    void add_map_point(MapPoint *map_point);

    void set_reference_map_points(const std::vector<MapPoint *> &map_points);
    void inform_new_big_change();
    int  get_last_big_change_index();

    long unsigned int map_points_in_map();
    long unsigned     key_frames_in_map();

    std::vector<KeyFrame *> get_all_key_frames();
    std::vector<MapPoint *> get_all_map_points();
    std::vector<MapPoint *> get_reference_map_points();

    std::vector<Map *> get_all_maps();

    int count_maps();

    void clear_map();

    void clear_atlas();

    Map *get_current_map();

    void set_map_bad(Map *map);

    bool is_inertial();
    void set_inertial_sensor();
    void set_imu_initialized();
    bool is_imu_initialized();

    std::map<long unsigned int, KeyFrame *> get_atlas_key_frames();

    long unsigned int get_num_lived_key_frames();

    long unsigned int get_num_lived_map_points();

  protected:
    void create_new_map_locked();

    std::set<Map *> _maps;
    // Retain maps removed from the active set while SLAM objects may still reference them.
    std::set<Map *> _owned_maps;
    Map            *_current_map{nullptr};

    unsigned long int _last_init_kf_id_map{0};

    std::mutex _mutex;
};

} // namespace ORB_SLAM3
