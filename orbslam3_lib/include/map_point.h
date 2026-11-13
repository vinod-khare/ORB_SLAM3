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

#ifndef MAP_POINT_H
#define MAP_POINT_H

#include "key_frame.h"
#include "converter.h"
#include "frame.h"
#include "map.h"

#include <map>
#include <mutex>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{

class key_frame;
class map;
class frame;

class map_point
{

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    map_point();

    map_point(const Eigen::Vector3f &Pos, key_frame *pRefKF, map *pMap);
    map_point(const double invDepth, cv::Point2f uv_init, key_frame *pRefKF, key_frame *pHostKF, map *pMap);
    map_point(const Eigen::Vector3f &Pos, map *pMap, frame *pFrame, const int &idxF, geometric_camera &camera);

    void            set_world_position(const Eigen::Vector3f &Pos);
    Eigen::Vector3f get_world_position();

    Eigen::Vector3f get_normal();
    void            set_normal_vector(const Eigen::Vector3f &normal);

    key_frame *get_reference_key_frame();

    std::map<key_frame *, std::tuple<int, int>> get_observations();
    int                                        get_observation_count();

    void                         add_observation(key_frame *pKF, int idx);
    std::array<unsigned char, 3> color();
    void                         erase_observation(key_frame *pKF);

    std::tuple<int, int> get_index_in_key_frame(key_frame *pKF);
    bool                 is_in_key_frame(key_frame *pKF);

    void set_bad_flag();
    bool is_bad();

    void       replace(map_point *pMP);
    map_point *get_replaced();

    void       increase_visible(int n = 1);
    void       increase_found(int n = 1);
    float      get_found_ratio();
    inline int get_found() { return found; }

    void compute_distinctive_descriptors();

    cv::Mat get_descriptor();

    void update_normal_and_depth();

    float get_min_distance_invariance();
    float get_max_distance_invariance();
    int   predict_scale(const float &currentDist, key_frame *pKF);
    int   predict_scale(const float &currentDist, frame *pF);

    map *get_map();
    void update_map(map *pMap);

    void print_observations();

  public:
    long unsigned int        id;
    static long unsigned int next_id;
    long int                 first_key_frame_id;
    long int                 first_frame_id;
    int                      observation_count;

    // Variables used by the tracking
    float             track_projection_x;
    float             track_projection_y;
    float             track_depth;
    float             track_depth_right;
    float             track_projection_x_right;
    float             track_projection_y_right;
    bool              track_in_view, track_in_view_right;
    int               track_scale_level, track_scale_level_right;
    float             track_view_cos, track_view_cos_right;
    long unsigned int track_reference_for_frame;
    long unsigned int last_frame_seen;

    // Variables used by local mapping
    long unsigned int local_ba_key_frame_id;
    long unsigned int fuse_candidate_key_frame_id;

    // Variables used by loop closing
    long unsigned int loop_point_key_frame_id;
    long unsigned int corrected_by_key_frame_id;
    long unsigned int corrected_reference_key_frame_id;
    Eigen::Vector3f   global_ba_position;
    long unsigned int global_ba_key_frame_id;
    long unsigned int local_ba_merge_key_frame_id;

    // Variable used by merging
    Eigen::Vector3f merge_position;
    Eigen::Vector3f merge_normal_vector;

    // Fopr inverse depth optimization
    double    inverse_depth;
    double    initial_u;
    double    initial_v;
    key_frame *host_key_frame;

    static std::mutex global_mutex;

    unsigned int origin_map_id;

  private:
    std::array<unsigned char, 3> _color{255, 255, 255};
    bool                         _has_color{false};

  protected:
    // Position in absolute coordinates
    Eigen::Vector3f world_position;

    // Keyframes observing the point and associated index in keyframe
    std::map<key_frame *, std::tuple<int, int>> observations_by_key_frame;

    // Mean viewing direction
    Eigen::Vector3f normal_vector;

    // Best descriptor to fast matching
    cv::Mat descriptor;

    // Reference key_frame
    key_frame *reference_key_frame;

    // Tracking counters
    int visible;
    int found;

    // Bad flag (we do not currently erase map_point from memory)
    bool       bad;
    map_point *replaced;

    // Scale invariance distances
    float minimum_distance;
    float maximum_distance;

    map *owner_map;

    // Mutex
    std::mutex position_mutex;
    std::mutex features_mutex;
    std::mutex map_mutex;
};

} // namespace ORB_SLAM3

#endif // MAP_POINT_H
