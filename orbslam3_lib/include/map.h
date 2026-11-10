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

#ifndef MAP_H
#define MAP_H

#include "KeyFrame.h"
#include "map_point.h"

#include <mutex>
#include <set>

namespace ORB_SLAM3
{

class map_point;
class KeyFrame;
class key_frame_database;

class map
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    map();
    map(int initKFid);
    ~map();

    void add_key_frame(KeyFrame *pKF);
    void add_map_point(map_point *pMP);
    void erase_map_point(map_point *pMP);
    void erase_key_frame(KeyFrame *pKF);
    void set_reference_map_points(const std::vector<map_point *> &vpMPs);
    void inform_new_big_change();
    int  get_last_big_change_index();

    std::vector<KeyFrame *>  get_all_key_frames();
    std::vector<map_point *> get_all_map_points();
    std::vector<map_point *> get_reference_map_points();

    long unsigned int map_point_count();
    long unsigned     key_frame_count();

    long unsigned int get_id();

    long unsigned int get_initial_key_frame_id();
    void              set_initial_key_frame_id(long unsigned int initKFif);
    long unsigned int get_max_key_frame_id();

    KeyFrame *get_origin_key_frame();

    void set_current_map();
    void set_stored_map();

    bool is_in_use();

    void set_bad();
    bool is_bad();

    void clear();

    int  get_map_change_index();
    void increase_change_index();
    int  get_last_map_change();
    void set_last_map_change(int currentChangeId);

    void set_imu_initialized();
    bool is_imu_initialized();

    void apply_scaled_rotation(const Sophus::SE3f &T, const float s, const bool bScaledVel = false);

    void set_inertial_sensor();
    bool is_inertial();
    void set_inertial_ba1();
    void set_inertial_ba2();
    bool get_inertial_ba1();
    bool get_inertial_ba2();

    void print_essential_graph();
    bool check_essential_graph();
    void change_id(long unsigned int nId);

    unsigned int get_lowest_key_frame_id();

    void print_reprojection_error(list<KeyFrame *> &lpLocalWindowKFs, KeyFrame *mpCurrentKF, string &name, string &name_folder);

    vector<KeyFrame *> key_frame_origins;
    KeyFrame          *first_region_key_frame;
    std::mutex         map_update_mutex;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex point_creation_mutex;

    bool failed;

    // Size of the thumbnail (always in power of 2)
    static const int thumb_width  = 512;
    static const int thumb_height = 512;

    static long unsigned int next_id;

    // DEBUG: show KFs which are used in LBA
    std::set<long unsigned int> optimized_key_frames;
    std::set<long unsigned int> fixed_key_frames;
    std::mutex                  viewer_mutex;

  protected:
    long unsigned int id;

    std::set<map_point *> map_points;
    std::set<KeyFrame *>  key_frames;

    KeyFrame *initial_key_frame;
    KeyFrame *lowest_id_key_frame;

    std::vector<map_point *> reference_map_points;

    bool imu_initialized;

    int map_change_index;
    int notified_map_change_index;

    long unsigned int initial_key_frame_id;
    long unsigned int max_key_frame_id;

    // Index related to a big change in the map (loop closure, global BA)
    int big_change_index;

    bool in_use;
    bool bad = false;

    bool inertial;
    bool imu_ba1;
    bool imu_ba2;

    // Mutex
    std::mutex map_mutex;
};

} // namespace ORB_SLAM3

#endif // MAP_H
