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

#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H

#include "loop_closing.h"
#include "tracking.h"
#include "atlas.h"
#include "key_frame.h"
#include "key_frame_database.h"
#include "processing_statistics.h"
#include "settings.h"

#include <mutex>

namespace ORB_SLAM3
{

class system;
class tracking;
class loop_closing;

class local_mapping
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    local_mapping(processing_statistics &statistics, system *pSys, atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName = std::string());

    void set_loop_closer(loop_closing *pLoopCloser);

    void set_tracker(tracking *pTracker);

    // Main function
    void run();

    void insert_key_frame(key_frame *pKF);
    void empty_queue();

    // Thread Synch
    void request_stop();
    void request_reset();
    void request_reset_active_map(map *pMap);
    bool stop();
    void release();
    bool is_stopped();
    bool stop_requested();
    bool accept_key_frames();
    void set_accept_key_frames(bool flag);
    bool set_not_stop(bool flag);

    void interrupt_bundle_adjustment();

    void request_finish();
    bool is_finished();

    int key_frames_in_queue()
    {
        unique_lock<std::mutex> lock(new_key_frames_mutex);
        return new_key_frames.size();
    }

    bool       is_initializing();
    double     get_current_key_frame_time();
    key_frame *get_current_key_frame();

    std::mutex imu_initialization_mutex;

    Eigen::MatrixXd inertial_covariance;
    Eigen::Matrix3d imu_world_rotation;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d accelerometer_bias;
    double          scale;
    double          initialization_time;
    double          initialization_cost_time;

    unsigned int initialization_section;
    unsigned int initialization_index;
    unsigned int key_frame_count;
    double       first_timestamp;
    int          inlier_match_count;

    // For debugging (erase in normal mode)
    int    initial_frame;
    int    iteration_index;
    string sequence_name;

    bool skip_bundle_adjustment_1;
    bool skip_bundle_adjustment_2;
    bool bad_imu;

    bool write_statistics;

    // not consider far points (clouds)
    bool  far_points_enabled;
    float far_point_threshold;

  protected:
    processing_statistics &_statistics;

    bool check_new_key_frames();
    void process_new_key_frame();
    void create_new_map_points();

    void cull_map_points();
    void search_in_neighbors();
    void cull_key_frames();

    system *owner_system;

    bool monocular;
    bool inertial;

    void       reset_if_requested();
    bool       reset_requested;
    bool       active_map_reset_requested;
    map       *map_to_reset;
    std::mutex reset_mutex;

    bool       check_finish();
    void       set_finish();
    bool       finish_requested;
    bool       finished;
    std::mutex finish_mutex;

    atlas *owner_atlas;

    loop_closing *loop_closer;
    tracking    *tracker;

    std::list<key_frame *> new_key_frames;

    key_frame *current_key_frame;

    std::list<map_point *> recently_added_map_points;

    std::mutex new_key_frames_mutex;

    bool bundle_adjustment_aborted;

    bool       stopped;
    bool       stop_requested_flag;
    bool       not_stop_requested;
    std::mutex stop_mutex;

    bool       accept_key_frames_flag;
    std::mutex accept_key_frames_mutex;

    void initialize_imu(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
    void refine_scale();

    bool initializing;

    Eigen::MatrixXd inertial_information;
    int             local_map_point_count;
    int             key_frame_culling_count;

    float initialization_elapsed_time;

    int scale_refinement_count;

    // DEBUG
    ofstream f_lm;
};

} // namespace ORB_SLAM3

#endif // LOCAL_MAPPING_H
