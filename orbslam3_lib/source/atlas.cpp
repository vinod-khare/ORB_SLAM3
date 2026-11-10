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

#include "atlas.h"

#include <algorithm>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

atlas::atlas(int initial_key_frame_id) : _last_init_kf_id_map(initial_key_frame_id) { create_new_map(); }

atlas::~atlas()
{
    for (map *map : _owned_maps)
    {
        delete map;
    }
}

void atlas::create_new_map()
{
    std::lock_guard<std::mutex> lock(_mutex);
    create_new_map_locked();
}

void atlas::create_new_map_locked()
{
    if (_current_map)
    {
        if (!_maps.empty() && _last_init_kf_id_map < _current_map->get_max_key_frame_id())
        {
            _last_init_kf_id_map = _current_map->get_max_key_frame_id() + 1; // The init KF is the next of current maximum
        }

        _current_map->set_stored_map();
    }

    _current_map = new map(_last_init_kf_id_map);
    _owned_maps.insert(_current_map);
    _current_map->set_current_map();
    _maps.insert(_current_map);
    spdlog::info("Created map {}", _current_map->get_id());
}

void atlas::change_map(map *map)
{
    if (!map)
    {
        throw std::invalid_argument("Cannot change to a null map");
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (!_maps.contains(map) || map->is_bad())
    {
        throw std::invalid_argument("Cannot change to a map that is not active in this atlas");
    }

    if (_current_map)
    {
        _current_map->set_stored_map();
    }

    _current_map = map;
    _current_map->set_current_map();
    spdlog::info("Changed to map {}", _current_map->get_id());
}

unsigned long int atlas::get_last_init_kf_id()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _last_init_kf_id_map;
}

void atlas::add_key_frame(KeyFrame *key_frame)
{
    map *map = key_frame->GetMap();
    map->add_key_frame(key_frame);
}

void atlas::add_map_point(map_point *map_point)
{
    map *map = map_point->get_map();
    map->add_map_point(map_point);
}

void atlas::set_reference_map_points(const std::vector<map_point *> &map_points)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _current_map->set_reference_map_points(map_points);
}

void atlas::inform_new_big_change()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _current_map->inform_new_big_change();
}

int atlas::get_last_big_change_index()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->get_last_big_change_index();
}

long unsigned int atlas::map_points_in_map()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->map_point_count();
}

long unsigned atlas::key_frames_in_map()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->key_frame_count();
}

std::vector<KeyFrame *> atlas::get_all_key_frames()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->get_all_key_frames();
}

std::vector<map_point *> atlas::get_all_map_points()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->get_all_map_points();
}

std::vector<map_point *> atlas::get_reference_map_points()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->get_reference_map_points();
}

std::vector<map *> atlas::get_all_maps()
{
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<map *>          maps(_maps.begin(), _maps.end());
    std::sort(maps.begin(), maps.end(), [](map *left, map *right) { return left->get_id() < right->get_id(); });
    return maps;
}

int atlas::count_maps()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return static_cast<int>(_maps.size());
}

void atlas::clear_map()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_current_map)
    {
        _current_map->clear();
    }
}

void atlas::clear_atlas()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _maps.clear();
    _current_map         = nullptr;
    _last_init_kf_id_map = 0;
}

map *atlas::get_current_map()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_current_map || _current_map->is_bad())
    {
        create_new_map_locked();
    }

    return _current_map;
}

void atlas::set_map_bad(map *map)
{
    if (!map)
    {
        throw std::invalid_argument("Cannot mark a null map as bad");
    }

    std::lock_guard<std::mutex> lock(_mutex);
    _owned_maps.insert(map);
    _maps.erase(map);
    map->set_bad();
}

bool atlas::is_inertial()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->is_inertial();
}

void atlas::set_inertial_sensor()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _current_map->set_inertial_sensor();
}

void atlas::set_imu_initialized()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _current_map->set_imu_initialized();
}

bool atlas::is_imu_initialized()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _current_map->is_imu_initialized();
}

long unsigned int atlas::get_num_lived_key_frames()
{
    std::lock_guard<std::mutex> lock(_mutex);
    long unsigned int           num = 0;
    for (map *map : _maps)
    {
        num += map->get_all_key_frames().size();
    }

    return num;
}

long unsigned int atlas::get_num_lived_map_points()
{
    std::lock_guard<std::mutex> lock(_mutex);
    long unsigned int           num = 0;
    for (map *map : _maps)
    {
        num += map->get_all_map_points().size();
    }

    return num;
}

std::map<long unsigned int, KeyFrame *> atlas::get_atlas_key_frames()
{
    std::lock_guard<std::mutex>             lock(_mutex);
    std::map<long unsigned int, KeyFrame *> key_frames_by_id;
    for (map *map : _maps)
    {
        std::vector<KeyFrame *> key_frames = map->get_all_key_frames();

        for (KeyFrame *key_frame : key_frames)
        {
            key_frames_by_id[key_frame->mnId] = key_frame;
        }
    }

    return key_frames_by_id;
}

} // namespace ORB_SLAM3
