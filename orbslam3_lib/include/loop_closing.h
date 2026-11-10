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

#ifndef LOOP_CLOSING_H
#define LOOP_CLOSING_H

#include "Tracking.h"
#include "atlas.h"
#include "key_frame.h"
#include "key_frame_database.h"
#include "local_mapping.h"
#include "place_vocabulary.h"
#include "processing_statistics.h"

#include <map>
#include <mutex>
#include <thread>

#include <boost/algorithm/string.hpp>

#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace ORB_SLAM3
{

class Tracking;
class local_mapping;
class key_frame_database;
class map;

class loop_closing
{
  public:
    typedef pair<set<key_frame *>, int>                                                                                                consistent_group;
    typedef std::map<key_frame *, g2o::Sim3, std::less<key_frame *>, Eigen::aligned_allocator<std::pair<key_frame *const, g2o::Sim3>>> key_frame_and_pose;

  public:
    loop_closing(processing_statistics &statistics, atlas *pAtlas, key_frame_database *pDB, place_vocabulary *pVoc, const bool bFixScale, const bool bActiveLC);
    ~loop_closing();

    void set_tracker(Tracking *pTracker);

    void set_local_mapper(local_mapping *pLocalMapper);

    // Main function
    void run();

    void insert_key_frame(key_frame *pKF);

    void request_reset();
    void request_reset_active_map(map *pMap);

    // This function will run in a separate thread
    void run_global_bundle_adjustment(map *pActiveMap, unsigned long nLoopKF);

    bool is_global_bundle_adjustment_running()
    {
        unique_lock<std::mutex> lock(global_bundle_adjustment_mutex);
        return global_bundle_adjustment_running;
    }
    bool is_global_bundle_adjustment_finished()
    {
        unique_lock<std::mutex> lock(global_bundle_adjustment_mutex);
        return global_bundle_adjustment_finished;
    }

    void request_finish();

    void stop_global_bundle_adjustment();

    bool is_finished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  protected:
    processing_statistics &_statistics;

    bool check_new_key_frames();

    // Methods to implement the new place recognition algorithm
    bool detect_new_common_regions();
    bool detect_and_refine_sim3_from_last_key_frame(
        key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs);
    bool detect_common_regions_from_bow(std::vector<key_frame *> &vpBowCand,
                                        key_frame               *&pMatchedKF,
                                        key_frame               *&pLastCurrentKF,
                                        g2o::Sim3                &g2oScw,
                                        int                      &nNumCoincidences,
                                        std::vector<map_point *> &vpMPs,
                                        std::vector<map_point *> &vpMatchedMPs);
    bool detect_common_regions_from_last_key_frame(
        key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs);
    int find_matches_by_projection(
        key_frame *pCurrentKF, key_frame *pMatchedKFw, g2o::Sim3 &g2oScw, set<map_point *> &spMatchedMPinOrigin, vector<map_point *> &vpMapPoints, vector<map_point *> &vpMatchedMapPoints);

    void search_and_fuse(const key_frame_and_pose &CorrectedPosesMap, vector<map_point *> &vpMapPoints);
    void search_and_fuse(const vector<key_frame *> &vConectedKFs, vector<map_point *> &vpMapPoints);

    void correct_loop();

    void merge_local();
    void merge_local_2();

    void check_observations(set<key_frame *> &spKFsMap1, set<key_frame *> &spKFsMap2);

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

    atlas    *owner_atlas;
    Tracking *tracker;

    key_frame_database *database;
    place_vocabulary   *place_vocabulary_ptr;

    local_mapping *local_mapper;

    std::list<key_frame *> loop_key_frame_queue;

    std::mutex loop_queue_mutex;

    // Loop detector parameters
    float covisibility_consistency_threshold;

    // Loop detector variables
    key_frame                    *current_key_frame;
    key_frame                    *last_current_key_frame;
    key_frame                    *matched_key_frame;
    std::vector<consistent_group> consistent_groups;
    std::vector<key_frame *>      enough_consistent_candidates;
    std::vector<key_frame *>      current_connected_key_frames;
    std::vector<map_point *>      current_matched_points;
    std::vector<map_point *>      loop_fusion_map_points;
    cv::Mat                       world_to_camera_sim3_matrix;
    g2o::Sim3                     world_to_camera_sim3;

    //-------
    map *last_map;

    bool                     loop_detected;
    int                      loop_coincidence_count;
    int                      loop_not_found_count;
    key_frame               *loop_last_current_key_frame;
    g2o::Sim3                loop_sim3_loop_to_world;
    g2o::Sim3                loop_sim3_world_to_camera;
    key_frame               *loop_matched_key_frame;
    std::vector<map_point *> loop_map_points;
    std::vector<map_point *> matched_map_points_in_loop;
    bool                     merge_detected;
    int                      merge_coincidence_count;
    int                      merge_not_found_count;
    key_frame               *merge_last_current_key_frame;
    g2o::Sim3                merge_sim3_loop_to_world;
    g2o::Sim3                merge_sim3_matched_to_world;
    g2o::Sim3                merge_sim3_world_to_camera;
    key_frame               *merge_matched_key_frame;
    std::vector<map_point *> merge_map_points;
    std::vector<map_point *> matched_map_points_in_merge;
    std::vector<key_frame *> merge_connected_key_frames;

    g2o::Sim3 old_to_new_similarity;
    //-------

    long unsigned int last_loop_key_frame_id;

    // Variables related to Global Bundle Adjustment
    bool         global_bundle_adjustment_running;
    bool         global_bundle_adjustment_finished;
    bool         global_bundle_adjustment_stop_requested;
    std::mutex   global_bundle_adjustment_mutex;
    std::jthread _global_ba_thread;

    // Fix scale in the stereo/RGB-D case
    bool fix_scale;

    int full_bundle_adjustment_index{};

    vector<double> current_candidate_times;
    vector<double> matched_candidate_times;
    vector<int>    recognition_types;

    // DEBUG
    string subtrajectory_folder;
    int    correction_count;
    int    global_ba_correction_count;

    // To (de)activate LC
    bool loop_closing_active = true;

#ifdef REGISTER_LOOP
    string loop_folder;
#endif
};

} // namespace ORB_SLAM3

#endif // LOOP_CLOSING_H
