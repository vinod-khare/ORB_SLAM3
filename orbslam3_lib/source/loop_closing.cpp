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

#include "loop_closing.h"

#include "converter.h"
#include "descriptor_matcher.h"
#include "g2o_types.h"
#include "optimizer.h"
#include "sim3_solver.h"

#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

loop_closing::loop_closing(processing_statistics &statistics, camera_view cameras, atlas *pAtlas, key_frame_database *pDB, place_vocabulary *pVoc, const bool bFixScale, const bool bActiveLC)
    : _statistics(statistics), _cameras(cameras), reset_requested(false), active_map_reset_requested(false), finish_requested(false), finished(true), owner_atlas(pAtlas), database(pDB),
      place_vocabulary_ptr(pVoc), matched_key_frame(NULL), loop_detected(false), loop_coincidence_count(0), loop_not_found_count(0), merge_detected(false), merge_coincidence_count(0),
      merge_not_found_count(0), last_loop_key_frame_id(0), global_bundle_adjustment_running(false), global_bundle_adjustment_finished(true), global_bundle_adjustment_stop_requested(false),
      fix_scale(bFixScale), full_bundle_adjustment_index(0), loop_closing_active(bActiveLC)
{
    covisibility_consistency_threshold = 3;
    last_current_key_frame             = static_cast<key_frame *>(NULL);
    subtrajectory_folder               = "SubTrajectories/";
    correction_count                   = 0;
    global_ba_correction_count         = 0;
}

loop_closing::~loop_closing() { stop_global_bundle_adjustment(); }

void loop_closing::set_tracker(tracking *pTracker) { tracker = pTracker; }

void loop_closing::set_local_mapper(local_mapping *pLocalMapper) { local_mapper = pLocalMapper; }

void loop_closing::run()
{
    finished = false;

    while (1)
    {

        // NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------

        if (check_new_key_frames())
        {
            const processing_statistics::backend_timer statistics_timer(_statistics, processing_statistics::backend_stage::loop_closing);
            if (last_current_key_frame)
            {
                last_current_key_frame->loop_candidate_key_frames.clear();
                last_current_key_frame->merge_candidate_key_frames.clear();
            }

            auto bFindedRegion = detect_new_common_regions();

            if (bFindedRegion)
            {
                if (merge_detected)
                {
                    if ((tracker->sensor == system::imu_monocular || tracker->sensor == system::imu_stereo || tracker->sensor == system::imu_rgbd) &&
                        !current_key_frame->get_map()->is_imu_initialized())
                    {
                        spdlog::warn("IMU is not initialized; aborting merge");
                    }
                    else
                    {
                        auto mTmw = merge_matched_key_frame->get_pose().cast<double>();
                        g2o::Sim3                  gSmw2(mTmw.unit_quaternion(), mTmw.translation(), 1.0);
                        auto mTcw = current_key_frame->get_pose().cast<double>();
                        g2o::Sim3                  gScw1(mTcw.unit_quaternion(), mTcw.translation(), 1.0);
                        auto gSw2c = merge_sim3_loop_to_world.inverse();
                        [[maybe_unused]] auto gSw1m = merge_sim3_loop_to_world;

                        old_to_new_similarity = gSw2c * gScw1;

                        if (current_key_frame->get_map()->is_inertial() && merge_matched_key_frame->get_map()->is_inertial())
                        {
                            spdlog::debug("Checking merge transformation with IMU");
                            if (old_to_new_similarity.scale() < 0.90 || old_to_new_similarity.scale() > 1.1)
                            {
                                merge_last_current_key_frame->set_erase();
                                merge_matched_key_frame->set_erase();
                                merge_coincidence_count = 0;
                                matched_map_points_in_merge.clear();
                                merge_map_points.clear();
                                merge_not_found_count = 0;
                                merge_detected        = false;
                                spdlog::warn("Scale estimate is outside merge bounds; aborting merge");
                                continue;
                            }
                            // If inertial, force only yaw
                            if ((tracker->sensor == system::imu_monocular || tracker->sensor == system::imu_stereo || tracker->sensor == system::imu_rgbd) &&
                                current_key_frame->get_map()->get_inertial_ba1())
                            {
                                auto phi   = log_so3(old_to_new_similarity.rotation().toRotationMatrix());
                                phi(0)                = 0;
                                phi(1)                = 0;
                                old_to_new_similarity = g2o::Sim3(exp_so3(phi), old_to_new_similarity.translation(), 1.0);
                            }
                        }

                        merge_sim3_matched_to_world = gSmw2 * gSw2c * gScw1;

                        merge_sim3_world_to_camera = merge_sim3_loop_to_world;

                        // tracker->set_step_by_step(true);

                        spdlog::info("Merge detected");

                        // TODO UNCOMMENT
                        if (tracker->sensor == system::imu_monocular || tracker->sensor == system::imu_stereo || tracker->sensor == system::imu_rgbd)
                        {
                            merge_local_2();
                        }
                        else
                        {
                            merge_local();
                        }

                        spdlog::info("Merge finished!");
                    }

                    current_candidate_times.push_back(current_key_frame->timestamp);
                    matched_candidate_times.push_back(merge_matched_key_frame->timestamp);
                    recognition_types.push_back(1);

                    // Reset all variables
                    merge_last_current_key_frame->set_erase();
                    merge_matched_key_frame->set_erase();
                    merge_coincidence_count = 0;
                    matched_map_points_in_merge.clear();
                    merge_map_points.clear();
                    merge_not_found_count = 0;
                    merge_detected        = false;

                    if (loop_detected)
                    {
                        // Reset Loop variables
                        loop_last_current_key_frame->set_erase();
                        loop_matched_key_frame->set_erase();
                        loop_coincidence_count = 0;
                        matched_map_points_in_loop.clear();
                        loop_map_points.clear();
                        loop_not_found_count = 0;
                        loop_detected        = false;
                    }
                }

                if (loop_detected)
                {
                    auto bGoodLoop = true;
                    current_candidate_times.push_back(current_key_frame->timestamp);
                    matched_candidate_times.push_back(loop_matched_key_frame->timestamp);
                    recognition_types.push_back(0);

                    spdlog::info("*Loop detected");

                    loop_sim3_world_to_camera = loop_sim3_loop_to_world; //*mvg2oSim3LoopTcw[nCurrentIndex];
                    if (current_key_frame->get_map()->is_inertial())
                    {
                        auto Twc = current_key_frame->get_pose_inverse().cast<double>();
                        g2o::Sim3    g2oTwc(Twc.unit_quaternion(), Twc.translation(), 1.0);
                        auto g2oSww_new = g2oTwc * loop_sim3_world_to_camera;

                        auto phi = log_so3(g2oSww_new.rotation().toRotationMatrix());
                        std::ostringstream phi_output;
                        phi_output << phi.transpose();
                        spdlog::debug("Loop rotation phi = {}", phi_output.str());
                        if (fabs(phi(0)) < 0.008f && fabs(phi(1)) < 0.008f && fabs(phi(2)) < 0.349f)
                        {
                            if (current_key_frame->get_map()->is_inertial())
                            {
                                // If inertial, force only yaw
                                if ((tracker->sensor == system::imu_monocular || tracker->sensor == system::imu_stereo || tracker->sensor == system::imu_rgbd) &&
                                    current_key_frame->get_map()->get_inertial_ba2())
                                {
                                    phi(0)                    = 0;
                                    phi(1)                    = 0;
                                    g2oSww_new                = g2o::Sim3(exp_so3(phi), g2oSww_new.translation(), 1.0);
                                    loop_sim3_world_to_camera = g2oTwc.inverse() * g2oSww_new;
                                }
                            }
                        }
                        else
                        {
                            spdlog::warn("Loop rejected because its rotation is out of bounds");
                            bGoodLoop = false;
                        }
                    }

                    if (bGoodLoop)
                    {

                        loop_fusion_map_points = loop_map_points;

                        correct_loop();

                        correction_count += 1;
                    }

                    // Reset all variables
                    loop_last_current_key_frame->set_erase();
                    loop_matched_key_frame->set_erase();
                    loop_coincidence_count = 0;
                    matched_map_points_in_loop.clear();
                    loop_map_points.clear();
                    loop_not_found_count = 0;
                    loop_detected        = false;
                }
            }
            last_current_key_frame = current_key_frame;
        }

        reset_if_requested();

        if (check_finish())
        {
            break;
        }

        usleep(5000);
    }

    set_finish();
}

void loop_closing::insert_key_frame(key_frame *pKF)
{
    unique_lock<mutex> lock(loop_queue_mutex);
    if (pKF->id != 0)
    {
        loop_key_frame_queue.push_back(pKF);
    }
}

bool loop_closing::check_new_key_frames()
{
    unique_lock<mutex> lock(loop_queue_mutex);
    return !loop_key_frame_queue.empty();
}

bool loop_closing::detect_new_common_regions()
{
    // To deactivate placerecognition. No loopclosing nor merging will be performed
    if (!loop_closing_active)
    {
        return false;
    }

    {
        unique_lock<mutex> lock(loop_queue_mutex);
        current_key_frame = loop_key_frame_queue.front();
        loop_key_frame_queue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        current_key_frame->set_not_erase();
        current_key_frame->current_place_recognition = true;

        last_map = current_key_frame->get_map();
    }

    if (last_map->is_inertial() && !last_map->get_inertial_ba2())
    {
        database->add(current_key_frame);
        current_key_frame->set_erase();
        return false;
    }

    if (tracker->sensor == system::stereo && last_map->get_all_key_frames().size() < 5) // 12
    {
        database->add(current_key_frame);
        current_key_frame->set_erase();
        return false;
    }

    if (last_map->get_all_key_frames().size() < 12)
    {
        database->add(current_key_frame);
        current_key_frame->set_erase();
        return false;
    }

    // Check the last candidates with geometric validation
    //  Loop candidates
    auto bLoopDetectedInKF = false;
    [[maybe_unused]] auto bCheckSpatial     = false;

    if (loop_coincidence_count > 0)
    {
        bCheckSpatial = true;
        // Find from the last KF candidates
        auto mTcl = (current_key_frame->get_pose() * loop_last_current_key_frame->get_pose_inverse()).cast<double>();
        g2o::Sim3           gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        auto gScw           = gScl * loop_sim3_loop_to_world;
        auto numProjMatches = 0;
        vector<map_point *> vpMatchedMPs;
        auto bCommonRegion = detect_and_refine_sim3_from_last_key_frame(current_key_frame, loop_matched_key_frame, gScw, numProjMatches, loop_map_points, vpMatchedMPs);
        if (bCommonRegion)
        {

            bLoopDetectedInKF = true;

            loop_coincidence_count++;
            loop_last_current_key_frame->set_erase();
            loop_last_current_key_frame = current_key_frame;
            loop_sim3_loop_to_world     = gScw;
            matched_map_points_in_loop  = vpMatchedMPs;

            loop_detected        = loop_coincidence_count >= 3;
            loop_not_found_count = 0;

            if (!loop_detected)
            {
                spdlog::debug("Place recognition loop detected with refined Sim3");
            }
        }
        else
        {
            bLoopDetectedInKF = false;

            loop_not_found_count++;
            if (loop_not_found_count >= 2)
            {
                loop_last_current_key_frame->set_erase();
                loop_matched_key_frame->set_erase();
                loop_coincidence_count = 0;
                matched_map_points_in_loop.clear();
                loop_map_points.clear();
                loop_not_found_count = 0;
            }
        }
    }

    // Merge candidates
    auto bMergeDetectedInKF = false;
    if (merge_coincidence_count > 0)
    {
        // Find from the last KF candidates
        auto mTcl = (current_key_frame->get_pose() * merge_last_current_key_frame->get_pose_inverse()).cast<double>();

        g2o::Sim3           gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        auto gScw           = gScl * merge_sim3_loop_to_world;
        auto numProjMatches = 0;
        vector<map_point *> vpMatchedMPs;
        auto bCommonRegion = detect_and_refine_sim3_from_last_key_frame(current_key_frame, merge_matched_key_frame, gScw, numProjMatches, merge_map_points, vpMatchedMPs);
        if (bCommonRegion)
        {
            bMergeDetectedInKF = true;

            merge_coincidence_count++;
            merge_last_current_key_frame->set_erase();
            merge_last_current_key_frame = current_key_frame;
            merge_sim3_loop_to_world     = gScw;
            matched_map_points_in_merge  = vpMatchedMPs;

            merge_detected = merge_coincidence_count >= 3;
        }
        else
        {
            merge_detected     = false;
            bMergeDetectedInKF = false;

            merge_not_found_count++;
            if (merge_not_found_count >= 2)
            {
                merge_last_current_key_frame->set_erase();
                merge_matched_key_frame->set_erase();
                merge_coincidence_count = 0;
                matched_map_points_in_merge.clear();
                merge_map_points.clear();
                merge_not_found_count = 0;
            }
        }
    }

    if (merge_detected || loop_detected)
    {
        database->add(current_key_frame);
        return true;
    }

    // TODO: This is only necessary if we use a minimun score for pick the best candidates
    const auto vpConnectedKeyFrames = current_key_frame->get_vector_covisible_key_frames();

    // Extract candidates from the bag of words
    vector<key_frame *> vpMergeBowCand, vpLoopBowCand;
    if (!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
        // Search in BoW
        database->detect_n_best_candidates(current_key_frame, vpLoopBowCand, vpMergeBowCand, 3);
    }

    // Check the BoW candidates if the geometric candidate list is empty
    // Loop candidates
    if (!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        loop_detected = detect_common_regions_from_bow(
            vpLoopBowCand, loop_matched_key_frame, loop_last_current_key_frame, loop_sim3_loop_to_world, loop_coincidence_count, loop_map_points, matched_map_points_in_loop);
    }
    // Merge candidates
    if (!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        merge_detected = detect_common_regions_from_bow(
            vpMergeBowCand, merge_matched_key_frame, merge_last_current_key_frame, merge_sim3_loop_to_world, merge_coincidence_count, merge_map_points, matched_map_points_in_merge);
    }

    database->add(current_key_frame);

    if (merge_detected || loop_detected)
    {
        return true;
    }

    current_key_frame->set_erase();
    current_key_frame->current_place_recognition = false;

    return false;
}

bool loop_closing::detect_and_refine_sim3_from_last_key_frame(
    key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs) const
{
    set<map_point *> spAlreadyMatchedMPs;
    nNumProjMatches = find_matches_by_projection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    const auto nProjMatches    = 30;
    const auto nProjOptMatches = 50;
    const auto nProjMatchesRep = 100;

    if (nNumProjMatches >= nProjMatches)
    {
        auto mTwm = pMatchedKF->get_pose_inverse().cast<double>();
        const g2o::Sim3                   gSwm(mTwm.unit_quaternion(), mTwm.translation(), 1.0);
        auto gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        auto bFixedScale = fix_scale; // TODO CHECK; Solo para el monocular inertial
        if (tracker->sensor == system::imu_monocular && !pCurrentKF->get_map()->get_inertial_ba2())
        {
            bFixedScale = false;
        }
        const auto numOptMatches = optimizer::optimize_sim3(_cameras, current_key_frame, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);

        if (numOptMatches > nProjOptMatches)
        {
            const g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(), 1.0);

            vector<map_point *> vpMatchedMP;
            vpMatchedMP.resize(current_key_frame->get_map_point_matches().size(), static_cast<map_point *>(NULL));

            nNumProjMatches = find_matches_by_projection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if (nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

bool loop_closing::detect_common_regions_from_bow(std::vector<key_frame *> &vpBowCand,
                                                  key_frame               *&pMatchedKF2,
                                                  key_frame               *&pLastCurrentKF,
                                                  g2o::Sim3                &g2oScw,
                                                  int                      &nNumCoincidences,
                                                  std::vector<map_point *> &vpMPs,
                                                  std::vector<map_point *> &vpMatchedMPs)
{
    auto nBoWMatches     = 20;
    auto nBoWInliers     = 15;
    auto nSim3Inliers    = 20;
    auto nProjMatches    = 50;
    auto nProjOptMatches = 80;

    auto spConnectedKeyFrames = current_key_frame->get_connected_key_frames();

    auto nNumCovisibles = 10;

    descriptor_matcher matcherBoW(0.9, true);
    descriptor_matcher matcher(0.75, true);

    // Varibles to select the best numbe
    key_frame               *pBestMatchedKF;
    auto nBestMatchesReproj   = 0;
    auto nBestNumCoindicendes = 0;
    g2o::Sim3                g2oBestScw;
    std::vector<map_point *> vpBestMapPoints;
    std::vector<map_point *> vpBestMatchedMapPoints;

    int         numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    auto index = 0;
    for (auto pKFi : vpBowCand)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        // Current KF against KF with covisibles version
        auto vpCovKFi = pKFi->get_best_covisibility_key_frames(nNumCovisibles);
        if (vpCovKFi.empty())
        {
            spdlog::debug("Covisible keyframe list is empty");
            vpCovKFi.push_back(pKFi);
        }
        else
        {
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }

        auto bAbortByNearKF = false;
        for (auto j = 0; std::cmp_less(j, vpCovKFi.size()); ++j)
        {
            if (spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                bAbortByNearKF = true;
                break;
            }
        }
        if (bAbortByNearKF)
        {
            continue;
        }

        std::vector<std::vector<map_point *>> vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<map_point *> spMatchedMPi;
        auto numBoWMatches = 0;

        auto pMostBoWMatchesKF  = pKFi;
        auto nMostBoWNumMatches = 0;

        auto vpMatchedPoints     = std::vector<map_point *>(current_key_frame->get_map_point_matches().size(), static_cast<map_point *>(NULL));
        auto vpKeyFrameMatchedMP = std::vector<key_frame *>(current_key_frame->get_map_point_matches().size(), static_cast<key_frame *>(NULL));

        [[maybe_unused]] auto nIndexMostBoWMatchesKF = 0;
        for (auto j = 0; std::cmp_less(j, vpCovKFi.size()); ++j)
        {
            if (!vpCovKFi[j] || vpCovKFi[j]->is_bad())
            {
                continue;
            }

            auto num = matcherBoW.search_by_vocabulary(current_key_frame, vpCovKFi[j], vvpMatchedMPs[j]);
            if (num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches     = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        for (auto j = 0; std::cmp_less(j, vpCovKFi.size()); ++j)
        {
            for (auto k = 0; std::cmp_less(k, vvpMatchedMPs[j].size()); ++k)
            {
                auto pMPi_j = vvpMatchedMPs[j][k];
                if (!pMPi_j || pMPi_j->is_bad())
                {
                    continue;
                }

                if (spMatchedMPi.find(pMPi_j) == spMatchedMPi.end())
                {
                    spMatchedMPi.insert(pMPi_j);
                    numBoWMatches++;

                    vpMatchedPoints[k]     = pMPi_j;
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j];
                }
            }
        }

        // pMostBoWMatchesKF = vpCovKFi[pMostBoWMatchesKF];

        if (numBoWMatches >= nBoWMatches) // TODO pick a good threshold
        {
            // Geometric validation
            auto bFixedScale = fix_scale;
            if (tracker->sensor == system::imu_monocular && !current_key_frame->get_map()->get_inertial_ba2())
            {
                bFixedScale = false;
            }

            auto solver = sim3_solver(*_cameras[0], current_key_frame, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.set_ransac_parameters(0.99, nBoWInliers, 300); // at least 15 inliers

            auto bNoMore = false;
            vector<bool>    vbInliers;
            int             nInliers;
            auto bConverge = false;
            Eigen::Matrix4f mTcm;
            while (!bConverge && !bNoMore)
            {
                mTcm = solver.iterate(20, bNoMore, vbInliers, nInliers, bConverge);
            }

            if (bConverge)
            {

                //  Match by reprojection
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->get_best_covisibility_key_frames(nNumCovisibles);
                vpCovKFi.push_back(pMostBoWMatchesKF);
                set<key_frame *> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                set<map_point *>    spMapPoints;
                vector<map_point *> vpMapPoints;
                vector<key_frame *> vpKeyFrames;
                for (auto pCovKFi : vpCovKFi)
                {
                    for (auto pCovMPij : pCovKFi->get_map_point_matches())
                    {
                        if (!pCovMPij || pCovMPij->is_bad())
                        {
                            continue;
                        }

                        if (spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                            vpKeyFrames.push_back(pCovKFi);
                        }
                    }
                }

                g2o::Sim3     gScm(solver.get_estimated_rotation().cast<double>(), solver.get_estimated_translation().cast<double>(), (double)solver.get_estimated_scale());
                g2o::Sim3     gSmw(pMostBoWMatchesKF->get_rotation().cast<double>(), pMostBoWMatchesKF->get_translation().cast<double>(), 1.0);
                auto gScw = gScm * gSmw; // Similarity matrix of current from the world position
                auto mScw = converter::to_sophus(gScw);

                vector<map_point *> vpMatchedMP;
                vpMatchedMP.resize(current_key_frame->get_map_point_matches().size(), static_cast<map_point *>(NULL));
                vector<key_frame *> vpMatchedKF;
                vpMatchedKF.resize(current_key_frame->get_map_point_matches().size(), static_cast<key_frame *>(NULL));
                auto numProjMatches = matcher.search_by_projection(_cameras, current_key_frame, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);

                if (numProjMatches >= nProjMatches)
                {
                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    [[maybe_unused]] auto bFixedScale = fix_scale;
                    if (tracker->sensor == system::imu_monocular && !current_key_frame->get_map()->get_inertial_ba2())
                    {
                        bFixedScale = false;
                    }

                    auto numOptMatches = optimizer::optimize_sim3(_cameras, current_key_frame, pKFi, vpMatchedMP, gScm, 10, fix_scale, mHessian7x7, true);

                    if (numOptMatches >= nSim3Inliers)
                    {
                        g2o::Sim3     gSmw(pMostBoWMatchesKF->get_rotation().cast<double>(), pMostBoWMatchesKF->get_translation().cast<double>(), 1.0);
                        auto gScw = gScm * gSmw; // Similarity matrix of current from the world position
                        auto mScw = converter::to_sophus(gScw);

                        vector<map_point *> vpMatchedMP;
                        vpMatchedMP.resize(current_key_frame->get_map_point_matches().size(), static_cast<map_point *>(NULL));
                        auto numProjOptMatches = matcher.search_by_projection(_cameras, current_key_frame, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);

                        if (numProjOptMatches >= nProjOptMatches)
                        {
                            auto max_x = -1, min_x = 1000000;
                            auto max_y = -1, min_y = 1000000;
                            for (auto pMPi : vpMatchedMP)
                            {
                                if (!pMPi || pMPi->is_bad())
                                {
                                    continue;
                                }

                                tuple<size_t, size_t> indexes = pMPi->get_index_in_key_frame(pKFi);
                                int                   index   = get<0>(indexes);
                                if (index >= 0)
                                {
                                    int coord_x = pKFi->keypoints[index].pt_undistorted.x;
                                    if (coord_x < min_x)
                                    {
                                        min_x = coord_x;
                                    }
                                    if (coord_x > max_x)
                                    {
                                        max_x = coord_x;
                                    }
                                    int coord_y = pKFi->keypoints[index].pt_undistorted.y;
                                    if (coord_y < min_y)
                                    {
                                        min_y = coord_y;
                                    }
                                    if (coord_y > max_y)
                                    {
                                        max_y = coord_y;
                                    }
                                }
                            }

                            auto nNumKFs = 0;
                            // vpMatchedMPs = vpMatchedMP;
                            // vpMPs = vpMapPoints;
                            //  Check the Sim3 transformation with the current key_frame covisibles
                            auto vpCurrentCovKFs = current_key_frame->get_best_covisibility_key_frames(nNumCovisibles);

                            auto j = 0;
                            while (nNumKFs < 3 && std::cmp_less(j, vpCurrentCovKFs.size()))
                            {
                                auto pKFj = vpCurrentCovKFs[j];
                                auto mTjc = (pKFj->get_pose() * current_key_frame->get_pose_inverse()).cast<double>();
                                g2o::Sim3           gSjc(mTjc.unit_quaternion(), mTjc.translation(), 1.0);
                                auto gSjw             = gSjc * gScw;
                                auto numProjMatches_j = 0;
                                vector<map_point *> vpMatchedMPs_j;
                                auto bValid = detect_common_regions_from_last_key_frame(pKFj, pMostBoWMatchesKF, gSjw, numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if (bValid)
                                {
                                    auto Tc_w        = current_key_frame->get_pose();
                                    auto Tw_cj       = pKFj->get_pose_inverse();
                                    auto Tc_cj       = Tc_w * Tw_cj;
                                    [[maybe_unused]] auto vector_dist = Tc_cj.translation();
                                    nNumKFs++;
                                }
                                j++;
                            }

                            if (nNumKFs < 3)
                            {
                                vnStage[index]        = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }

                            if (nBestMatchesReproj < numProjOptMatches)
                            {
                                nBestMatchesReproj     = numProjOptMatches;
                                nBestNumCoindicendes   = nNumKFs;
                                pBestMatchedKF         = pMostBoWMatchesKF;
                                g2oBestScw             = gScw;
                                vpBestMapPoints        = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP;
                            }
                        }
                    }
                }
            }
        }
        index++;
    }

    if (nBestMatchesReproj > 0)
    {
        pLastCurrentKF   = current_key_frame;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF2      = pBestMatchedKF;
        pMatchedKF2->set_not_erase();
        g2oScw       = g2oBestScw;
        vpMPs        = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        return nNumCoincidences >= 3;
    }
    else
    {
        auto maxStage = -1;
        [[maybe_unused]] int maxMatched;
        for (auto i = 0; std::cmp_less(i, vnStage.size()); ++i)
        {
            if (vnStage[i] > maxStage)
            {
                maxStage   = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
    }
    return false;
}

bool loop_closing::detect_common_regions_from_last_key_frame(
    key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs) const
{
    set<map_point *> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = find_matches_by_projection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    const auto nProjMatches = 30;
    if (nNumProjMatches >= nProjMatches)
    {
        return true;
    }

    return false;
}

int loop_closing::find_matches_by_projection(
    key_frame *pCurrentKF, key_frame *pMatchedKFw, const g2o::Sim3 &g2oScw, set<map_point *> &spMatchedMPinOrigin, vector<map_point *> &vpMapPoints, vector<map_point *> &vpMatchedMapPoints) const
{
    const auto nNumCovisibles = 10;
    auto vpCovKFm       = pMatchedKFw->get_best_covisibility_key_frames(nNumCovisibles);
    const int                 nInitialCov    = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<key_frame *> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    auto spCurrentCovisbles = pCurrentKF->get_connected_key_frames();
    if (nInitialCov < nNumCovisibles)
    {
        for (auto i = 0; i < nInitialCov; ++i)
        {
            auto vpKFs     = vpCovKFm[i]->get_best_covisibility_key_frames(nNumCovisibles);
            auto nInserted = 0;
            auto j         = 0;
            while (std::cmp_less(j, vpKFs.size()) && nInserted < nNumCovisibles)
            {
                if (spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end())
                {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }
    set<map_point *> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    for (const auto pKFi : vpCovKFm)
    {
        for (auto pMPij : pKFi->get_map_point_matches())
        {
            if (!pMPij || pMPij->is_bad())
            {
                continue;
            }

            if (spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    auto mScw = converter::to_sophus(g2oScw);
    descriptor_matcher matcher(0.9, true);

    vpMatchedMapPoints.resize(pCurrentKF->get_map_point_matches().size(), static_cast<map_point *>(NULL));
    const auto num_matches = matcher.search_by_projection(_cameras, pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

void loop_closing::correct_loop()
{

    // If a Global Bundle Adjustment is running, abort it
    if (is_global_bundle_adjustment_running())
    {
        spdlog::info("Stopping global bundle adjustment");
        stop_global_bundle_adjustment();
        spdlog::info("Global bundle adjustment stopped");
    }

    local_mapper->request_stop();
    local_mapper->empty_queue();

    // Wait until Local Mapping has effectively stopped
    while (!local_mapper->is_stopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    current_key_frame->update_connections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    current_connected_key_frames = current_key_frame->get_vector_covisible_key_frames();
    current_connected_key_frames.push_back(current_key_frame);

    key_frame_and_pose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[current_key_frame] = loop_sim3_world_to_camera;
    auto Twc                 = current_key_frame->get_pose_inverse();
    auto Tcw                 = current_key_frame->get_pose();
    g2o::Sim3    g2oScw(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>(), 1.0);
    NonCorrectedSim3[current_key_frame] = g2oScw;

    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
    Sophus::SE3d correctedTcw(loop_sim3_world_to_camera.rotation(), loop_sim3_world_to_camera.translation() / loop_sim3_world_to_camera.scale());
    current_key_frame->set_pose(correctedTcw.cast<float>());

    auto pLoopMap = current_key_frame->get_map();

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->map_update_mutex);

        const auto bImuInit = pLoopMap->is_imu_initialized();

        for (auto vit = current_connected_key_frames.begin(), vend = current_connected_key_frames.end(); vit != vend; vit++)
        {
            auto pKFi = *vit;

            if (pKFi != current_key_frame)
            {
                auto Tiw = pKFi->get_pose();
                auto Tic = (Tiw * Twc).cast<double>();
                g2o::Sim3    g2oSic(Tic.unit_quaternion(), Tic.translation(), 1.0);
                auto g2oCorrectedSiw = g2oSic * loop_sim3_world_to_camera;
                // Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi] = g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
                pKFi->set_pose(correctedTiw.cast<float>());

                // Pose without correction
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion().cast<double>(), Tiw.translation().cast<double>(), 1.0);
                NonCorrectedSim3[pKFi] = g2oSiw;
            }
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for (auto mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
        {
            auto pKFi            = mit->first;
            auto g2oCorrectedSiw = mit->second;
            auto g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            auto g2oSiw = NonCorrectedSim3[pKFi];

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)

            auto vpMPsi = pKFi->get_map_point_matches();
            for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
            {
                auto pMPi = vpMPsi[iMP];
                if (!pMPi)
                {
                    continue;
                }
                if (pMPi->is_bad())
                {
                    continue;
                }
                if (pMPi->corrected_by_key_frame_id == current_key_frame->id)
                {
                    continue;
                }

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw             = pMPi->get_world_position().cast<double>();
                auto eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw));

                pMPi->set_world_position(eigCorrectedP3Dw.cast<float>());
                pMPi->corrected_by_key_frame_id        = current_key_frame->id;
                pMPi->corrected_reference_key_frame_id = pKFi->id;
                pMPi->update_normal_and_depth(*_cameras[0]);
            }

            // Correct velocity according to orientation correction
            if (bImuInit)
            {
                auto Rcor = (g2oCorrectedSiw.rotation().inverse() * g2oSiw.rotation()).cast<float>();
                pKFi->set_velocity(Rcor * pKFi->get_velocity());
            }

            // Make sure connections are updated
            pKFi->update_connections();
        }
        // TODO Check this index increasement
        owner_atlas->get_current_map()->increase_change_index();

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; std::cmp_less(i, matched_map_points_in_loop.size()); i++)
        {
            if (matched_map_points_in_loop[i])
            {
                auto pLoopMP = matched_map_points_in_loop[i];
                auto pCurMP  = current_key_frame->get_map_point(i);
                if (pCurMP)
                {
                    pCurMP->replace(pLoopMP);
                }
                else
                {
                    current_key_frame->add_map_point(pLoopMP, i);
                    pLoopMP->add_observation(current_key_frame, i);
                    pLoopMP->compute_distinctive_descriptors();
                }
            }
        }
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    search_and_fuse(CorrectedSim3, loop_fusion_map_points);

    // After the map_point fusion, new links in the covisibility graph will appear attaching both sides of the loop
    std::map<key_frame *, set<key_frame *>> LoopConnections;

    for (auto vit = current_connected_key_frames.begin(), vend = current_connected_key_frames.end(); vit != vend; vit++)
    {
        auto pKFi                = *vit;
        auto vpPreviousNeighbors = pKFi->get_vector_covisible_key_frames();

        // Update connections. Detect new links.
        pKFi->update_connections();
        LoopConnections[pKFi] = pKFi->get_connected_key_frames();
        for (auto vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for (auto vit2 = current_connected_key_frames.begin(), vend2 = current_connected_key_frames.end(); vit2 != vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    auto bFixedScale = fix_scale;
    // TODO CHECK; Solo para el monocular inertial
    if (tracker->sensor == system::imu_monocular && !current_key_frame->get_map()->get_inertial_ba2())
    {
        bFixedScale = false;
    }

    if (pLoopMap->is_inertial() && pLoopMap->is_imu_initialized())
    {
        optimizer::optimize_essential_graph_4dof(_cameras, pLoopMap, loop_matched_key_frame, current_key_frame, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }
    else
    {
        optimizer::optimize_essential_graph(_cameras, pLoopMap, loop_matched_key_frame, current_key_frame, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale);
    }

    owner_atlas->inform_new_big_change();

    // Add loop edge
    loop_matched_key_frame->add_loop_edge(current_key_frame);
    current_key_frame->add_loop_edge(loop_matched_key_frame);

    // Launch a new thread to perform Global Bundle Adjustment (Only if few keyframes, if not it would take too much time)
    if (!pLoopMap->is_imu_initialized() || (pLoopMap->key_frame_count() < 200 && owner_atlas->count_maps() == 1))
    {
        stop_global_bundle_adjustment();
        global_bundle_adjustment_running        = true;
        global_bundle_adjustment_finished       = false;
        global_bundle_adjustment_stop_requested = false;
        global_ba_correction_count              = correction_count;

        _global_ba_thread = std::jthread(&loop_closing::run_global_bundle_adjustment, this, pLoopMap, current_key_frame->id);
    }

    // Loop closed. Release Local Mapping.
    local_mapper->release();

    last_loop_key_frame_id = current_key_frame->id; // TODO old varible, it is not use in the new algorithm
}

void loop_closing::merge_local()
{
    auto numTemporalKFs = 25; // Temporal KFs in the local window if the map is inertial.

    // Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    key_frame *pNewChild;
    key_frame *pNewParent;

    vector<key_frame *> vpLocalCurrentWindowKFs;
    vector<key_frame *> vpMergeConnectedKFs;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    auto bRelaunchBA = false;

    //  If a Global Bundle Adjustment is running, abort it
    if (is_global_bundle_adjustment_running())
    {
        stop_global_bundle_adjustment();
        bRelaunchBA = true;
    }

    local_mapper->request_stop();
    // Wait until Local Mapping has effectively stopped
    while (!local_mapper->is_stopped())
    {
        usleep(1000);
    }

    local_mapper->empty_queue();

    // Merge map will become in the new active map with the local window of KFs and MPs from the current map.
    // Later, the elements of the current map will be transform to the new active map reference, in order to keep real time tracking
    auto pCurrentMap = current_key_frame->get_map();
    auto pMergeMap   = merge_matched_key_frame->get_map();

    // Ensure current keyframe is updated
    current_key_frame->update_connections();

    // Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    set<key_frame *> spLocalWindowKFs;
    // Get MPs in the welding area from the current map
    set<map_point *> spLocalWindowMPs;
    if (pCurrentMap->is_inertial() && pMergeMap->is_inertial()) // TODO Check the correct initialization
    {
        auto pKFi      = current_key_frame;
        auto nInserted = 0;
        while (pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            pKFi = current_key_frame->previous_key_frame;
            nInserted++;

            auto spMPi = pKFi->get_map_points();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = current_key_frame->next_key_frame;
        while (pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            auto spMPi = pKFi->get_map_points();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());

            pKFi = current_key_frame->next_key_frame;
        }
    }
    else
    {
        spLocalWindowKFs.insert(current_key_frame);
    }

    auto vpCovisibleKFs = current_key_frame->get_best_covisibility_key_frames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spLocalWindowKFs.insert(current_key_frame);
    constexpr auto nMaxTries = 5;
    auto nNumTries = 0;
    while (std::cmp_less(spLocalWindowKFs.size(), numTemporalKFs) && nNumTries < nMaxTries)
    {
        vector<key_frame *> vpNewCovKFs;
        vpNewCovKFs.clear();
        for (auto pKFi : spLocalWindowKFs)
        {
            auto vpKFiCov = pKFi->get_best_covisibility_key_frames(numTemporalKFs / 2);
            for (auto pKFcov : vpKFiCov)
            {
                if (pKFcov && !pKFcov->is_bad() && spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }
            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    for (auto pKFi : spLocalWindowKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        auto spMPs = pKFi->get_map_points();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }

    set<key_frame *> spMergeConnectedKFs;
    if (pCurrentMap->is_inertial() && pMergeMap->is_inertial()) // TODO Check the correct initialization
    {
        auto pKFi      = merge_matched_key_frame;
        auto nInserted = 0;
        while (pKFi && nInserted < numTemporalKFs / 2)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = current_key_frame->previous_key_frame;
            nInserted++;
        }

        pKFi = merge_matched_key_frame->next_key_frame;
        while (pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = current_key_frame->next_key_frame;
        }
    }
    else
    {
        spMergeConnectedKFs.insert(merge_matched_key_frame);
    }
    vpCovisibleKFs = merge_matched_key_frame->get_best_covisibility_key_frames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spMergeConnectedKFs.insert(merge_matched_key_frame);
    nNumTries = 0;
    while (std::cmp_less(spMergeConnectedKFs.size(), numTemporalKFs) && nNumTries < nMaxTries)
    {
        vector<key_frame *> vpNewCovKFs;
        for (auto pKFi : spMergeConnectedKFs)
        {
            auto vpKFiCov = pKFi->get_best_covisibility_key_frames(numTemporalKFs / 2);
            for (auto pKFcov : vpKFiCov)
            {
                if (pKFcov && !pKFcov->is_bad() && spMergeConnectedKFs.find(pKFcov) == spMergeConnectedKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }
            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    set<map_point *> spMapPointMerge;
    for (auto pKFi : spMergeConnectedKFs)
    {
        auto vpMPs = pKFi->get_map_points();
        spMapPointMerge.insert(vpMPs.begin(), vpMPs.end());
    }

    vector<map_point *> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    //
    auto Twc = current_key_frame->get_pose_inverse().cast<double>();
    g2o::Sim3    g2oNonCorrectedSwc(Twc.unit_quaternion(), Twc.translation(), 1.0);
    auto g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    auto g2oCorrectedScw    = merge_sim3_world_to_camera; // TODO Check the transformation

    key_frame_and_pose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[current_key_frame]    = g2oCorrectedScw;
    vNonCorrectedSim3[current_key_frame] = g2oNonCorrectedScw;

    for (auto pKFi : spLocalWindowKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            spdlog::trace("Bad KF in correction");
            continue;
        }

        if (pKFi->get_map() != pCurrentMap)
        {
            spdlog::trace("Other map KF, this should't happen");
        }

        g2o::Sim3 g2oCorrectedSiw;

        if (pKFi != current_key_frame)
        {
            auto Tiw = pKFi->get_pose().cast<double>();
            g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
            // Pose without correction
            vNonCorrectedSim3[pKFi] = g2oSiw;

            auto Tic = Tiw * Twc;
            g2o::Sim3    g2oSic(Tic.unit_quaternion(), Tic.translation(), 1.0);
            g2oCorrectedSiw      = g2oSic * merge_sim3_world_to_camera;
            vCorrectedSim3[pKFi] = g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->camera_pose_merge = pKFi->get_pose();

        // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
        auto s    = g2oCorrectedSiw.scale();
        pKFi->scale = s;
        Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

        pKFi->camera_pose_merge = correctedTiw.cast<float>();

        if (pCurrentMap->is_imu_initialized())
        {
            auto Rcor    = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
            pKFi->velocity_world_merge = Rcor * pKFi->get_velocity();
        }

        // TODO DEBUG to know which are the KFs that had been moved to the other map
    }

    auto numPointsWithCorrection = 0;

    // for(map_point* pMPi : spLocalWindowMPs)
    auto itMP = spLocalWindowMPs.begin();
    while (itMP != spLocalWindowMPs.end())
    {
        auto pMPi = *itMP;
        if (!pMPi || pMPi->is_bad())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }

        auto pKFref = pMPi->get_reference_key_frame();
        if (vCorrectedSim3.find(pKFref) == vCorrectedSim3.end())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            numPointsWithCorrection++;
            continue;
        }
        auto g2oCorrectedSwi    = vCorrectedSim3[pKFref].inverse();
        auto g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        Eigen::Vector3d    P3Dw             = pMPi->get_world_position().cast<double>();
        auto eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
        auto Rcor             = g2oCorrectedSwi.rotation() * g2oNonCorrectedSiw.rotation();

        pMPi->merge_position      = eigCorrectedP3Dw.cast<float>();
        pMPi->merge_normal_vector = Rcor.cast<float>() * pMPi->get_normal();

        itMP++;
    }

    {
        unique_lock<mutex> currentLock(pCurrentMap->map_update_mutex); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->map_update_mutex);     // We remove the Kfs and MPs in the merged area from the old map

        for (auto pKFi : spLocalWindowKFs)
        {
            if (!pKFi || pKFi->is_bad())
            {
                continue;
            }

            pKFi->camera_pose_before_merge = pKFi->get_pose();
            pKFi->world_pose_before_merge  = pKFi->get_pose_inverse();
            pKFi->set_pose(pKFi->camera_pose_merge);

            // Make sure connections are updated
            pKFi->update_map(pMergeMap);
            pKFi->merge_corrected_key_frame_id = current_key_frame->id;
            pMergeMap->add_key_frame(pKFi);
            pCurrentMap->erase_key_frame(pKFi);

            if (pCurrentMap->is_imu_initialized())
            {
                pKFi->set_velocity(pKFi->velocity_world_merge);
            }
        }

        for (auto pMPi : spLocalWindowMPs)
        {
            if (!pMPi || pMPi->is_bad())
            {
                continue;
            }

            pMPi->set_world_position(pMPi->merge_position);
            pMPi->set_normal_vector(pMPi->merge_normal_vector);
            pMPi->update_map(pMergeMap);
            pMergeMap->add_map_point(pMPi);
            pCurrentMap->erase_map_point(pMPi);
        }

        owner_atlas->change_map(pMergeMap);
        owner_atlas->set_map_bad(pCurrentMap);
        pMergeMap->increase_change_index();
        // TODO for debug
        pMergeMap->change_id(pCurrentMap->get_id());
    }

    // Rebuild the essential graph in the local window
    pCurrentMap->get_origin_key_frame()->set_first_connection(false);
    pNewChild  = current_key_frame->get_parent(); // Old parent, it will be the new child of this KF
    pNewParent = current_key_frame;               // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    current_key_frame->change_parent(merge_matched_key_frame);
    while (pNewChild)
    {
        pNewChild->erase_child(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        auto pOldParent = pNewChild->get_parent();

        pNewChild->change_parent(pNewParent);

        pNewParent = pNewChild;
        pNewChild  = pOldParent;
    }

    // Update the connections between the local window
    merge_matched_key_frame->update_connections();

    vpMergeConnectedKFs = merge_matched_key_frame->get_vector_covisible_key_frames();
    vpMergeConnectedKFs.push_back(merge_matched_key_frame);

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    search_and_fuse(vCorrectedSim3, vpCheckFuseMapPoint);

    // Update connectivity
    for (auto pKFi : spLocalWindowKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        pKFi->update_connections();
    }
    for (auto pKFi : spMergeConnectedKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        pKFi->update_connections();
    }

    auto bStop = false;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    if (tracker->sensor == system::imu_monocular || tracker->sensor == system::imu_stereo || tracker->sensor == system::imu_rgbd)
    {
        optimizer::merge_inertial_ba(_cameras, current_key_frame, merge_matched_key_frame, &bStop, pCurrentMap, vCorrectedSim3);
    }
    else
    {
        optimizer::local_bundle_adjustment(_cameras, current_key_frame, vpLocalCurrentWindowKFs, vpMergeConnectedKFs, &bStop);
    }

    // Loop closed. Release Local Mapping.
    local_mapper->release();

    // Update the non critical area from the current map to the merged map
    auto vpCurrentMapKFs = pCurrentMap->get_all_key_frames();
    auto vpCurrentMapMPs = pCurrentMap->get_all_map_points();

    if (vpCurrentMapKFs.size() == 0)
    {
    }
    else
    {
        if (tracker->sensor == system::monocular)
        {
            unique_lock<mutex> currentLock(pCurrentMap->map_update_mutex); // We update the current map with the Merge information

            for (auto pKFi : vpCurrentMapKFs)
            {
                if (!pKFi || pKFi->is_bad() || pKFi->get_map() != pCurrentMap)
                {
                    continue;
                }

                g2o::Sim3 g2oCorrectedSiw;

                auto Tiw = pKFi->get_pose().cast<double>();
                g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
                // Pose without correction
                vNonCorrectedSim3[pKFi] = g2oSiw;

                auto Tic = Tiw * Twc;
                g2o::Sim3    g2oSim(Tic.unit_quaternion(), Tic.translation(), 1.0);
                g2oCorrectedSiw      = g2oSim * merge_sim3_world_to_camera;
                vCorrectedSim3[pKFi] = g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                auto s = g2oCorrectedSiw.scale();

                pKFi->scale = s;

                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

                pKFi->camera_pose_before_merge = pKFi->get_pose();
                pKFi->world_pose_before_merge  = pKFi->get_pose_inverse();

                pKFi->set_pose(correctedTiw.cast<float>());

                if (pCurrentMap->is_imu_initialized())
                {
                    auto Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
                    pKFi->set_velocity(Rcor * pKFi->get_velocity()); // TODO: should add here scale s
                }
            }
            for (auto pMPi : vpCurrentMapMPs)
            {
                if (!pMPi || pMPi->is_bad() || pMPi->get_map() != pCurrentMap)
                {
                    continue;
                }

                auto pKFref             = pMPi->get_reference_key_frame();
                auto g2oCorrectedSwi    = vCorrectedSim3[pKFref].inverse();
                auto g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw             = pMPi->get_world_position().cast<double>();
                auto eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
                pMPi->set_world_position(eigCorrectedP3Dw.cast<float>());

                pMPi->update_normal_and_depth(*_cameras[0]);
            }
        }

        local_mapper->request_stop();
        // Wait until Local Mapping has effectively stopped
        while (!local_mapper->is_stopped())
        {
            usleep(1000);
        }

        // Optimize graph (and update the loop position for each element form the begining to the end)
        if (tracker->sensor != system::monocular)
        {
            optimizer::optimize_essential_graph(_cameras, current_key_frame, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }

        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->map_update_mutex); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->map_update_mutex);     // We remove the Kfs and MPs in the merged area from the old map

            for (auto pKFi : vpCurrentMapKFs)
            {
                if (!pKFi || pKFi->is_bad() || pKFi->get_map() != pCurrentMap)
                {
                    continue;
                }

                // Make sure connections are updated
                pKFi->update_map(pMergeMap);
                pMergeMap->add_key_frame(pKFi);
                pCurrentMap->erase_key_frame(pKFi);
            }

            for (auto pMPi : vpCurrentMapMPs)
            {
                if (!pMPi || pMPi->is_bad())
                {
                    continue;
                }

                pMPi->update_map(pMergeMap);
                pMergeMap->add_map_point(pMPi);
                pCurrentMap->erase_map_point(pMPi);
            }
        }
    }

    local_mapper->release();

    if (bRelaunchBA && (!pCurrentMap->is_imu_initialized() || (pCurrentMap->key_frame_count() < 200 && owner_atlas->count_maps() == 1)))
    {
        // Launch a new thread to perform Global Bundle Adjustment
        stop_global_bundle_adjustment();
        global_bundle_adjustment_running        = true;
        global_bundle_adjustment_finished       = false;
        global_bundle_adjustment_stop_requested = false;
        _global_ba_thread                       = std::jthread(&loop_closing::run_global_bundle_adjustment, this, pMergeMap, current_key_frame->id);
    }

    merge_matched_key_frame->add_merge_edge(current_key_frame);
    current_key_frame->add_merge_edge(merge_matched_key_frame);

    pCurrentMap->increase_change_index();
    pMergeMap->increase_change_index();
}

void loop_closing::merge_local_2()
{

    [[maybe_unused]] auto numTemporalKFs = 11; // TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    // Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    key_frame *pNewChild;
    key_frame *pNewParent;

    vector<key_frame *> vpLocalCurrentWindowKFs;
    vector<key_frame *> vpMergeConnectedKFs;

    key_frame_and_pose CorrectedSim3, NonCorrectedSim3;
    // NonCorrectedSim3[current_key_frame]=loop_sim3_world_to_camera;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    [[maybe_unused]] auto bRelaunchBA = false;

    //  If a Global Bundle Adjustment is running, abort it
    if (is_global_bundle_adjustment_running())
    {
        stop_global_bundle_adjustment();
        bRelaunchBA = true;
    }

    local_mapper->request_stop();
    // Wait until Local Mapping has effectively stopped
    while (!local_mapper->is_stopped())
    {
        usleep(1000);
    }

    auto pCurrentMap = current_key_frame->get_map();
    auto pMergeMap   = merge_matched_key_frame->get_map();

    {
        float        s_on = old_to_new_similarity.scale();
        Sophus::SE3f T_on(old_to_new_similarity.rotation().cast<float>(), old_to_new_similarity.translation().cast<float>());

        unique_lock<mutex> lock(owner_atlas->get_current_map()->map_update_mutex);

        local_mapper->empty_queue();

        auto bScaleVel = false;
        if (s_on != 1)
        {
            bScaleVel = true;
        }
        owner_atlas->get_current_map()->apply_scaled_rotation(*_cameras[0], T_on, s_on, bScaleVel);
        tracker->update_frame_imu(s_on, current_key_frame->get_imu_bias(), tracker->get_last_key_frame());
    }

    const int numKFnew = pCurrentMap->key_frame_count();

    if ((tracker->sensor == system::imu_monocular || tracker->sensor == system::imu_stereo || tracker->sensor == system::imu_rgbd) && !pCurrentMap->get_inertial_ba2())
    {
        // Map is not completly initialized
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.;
        ba << 0., 0., 0.;
        optimizer::inertial_optimization(_cameras, pCurrentMap, bg, ba);
        imu::bias          b(ba[0], ba[1], ba[2], bg[0], bg[1], bg[2]);
        unique_lock<mutex> lock(owner_atlas->get_current_map()->map_update_mutex);
        tracker->update_frame_imu(1.0f, b, tracker->get_last_key_frame());

        // Set map initialized
        pCurrentMap->set_inertial_ba2();
        pCurrentMap->set_inertial_ba1();
        pCurrentMap->set_imu_initialized();
    }

    // Load KFs and MPs from merge map
    {
        // Get Merge Map Mutex (This section stops tracking!!)
        unique_lock<mutex> currentLock(pCurrentMap->map_update_mutex); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->map_update_mutex);     // We remove the Kfs and MPs in the merged area from the old map

        auto vpMergeMapKFs = pMergeMap->get_all_key_frames();
        auto vpMergeMapMPs = pMergeMap->get_all_map_points();

        for (auto pKFi : vpMergeMapKFs)
        {
            if (!pKFi || pKFi->is_bad() || pKFi->get_map() != pMergeMap)
            {
                continue;
            }

            // Make sure connections are updated
            pKFi->update_map(pCurrentMap);
            pCurrentMap->add_key_frame(pKFi);
            pMergeMap->erase_key_frame(pKFi);
        }

        for (auto pMPi : vpMergeMapMPs)
        {
            if (!pMPi || pMPi->is_bad() || pMPi->get_map() != pMergeMap)
            {
                continue;
            }

            pMPi->update_map(pCurrentMap);
            pCurrentMap->add_map_point(pMPi);
            pMergeMap->erase_map_point(pMPi);
        }

        // Save non corrected poses (already merged maps)
        auto vpKFs = pCurrentMap->get_all_key_frames();
        for (auto pKFi : vpKFs)
        {
            auto Tiw = pKFi->get_pose().cast<double>();
            g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
            NonCorrectedSim3[pKFi] = g2oSiw;
        }
    }

    // Critical zone

    pMergeMap->get_origin_key_frame()->set_first_connection(false);
    pNewChild  = merge_matched_key_frame->get_parent(); // Old parent, it will be the new child of this KF
    pNewParent = merge_matched_key_frame;               // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    merge_matched_key_frame->change_parent(current_key_frame);
    while (pNewChild)
    {
        pNewChild->erase_child(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        auto pOldParent = pNewChild->get_parent();
        pNewChild->change_parent(pNewParent);
        pNewParent = pNewChild;
        pNewChild  = pOldParent;
    }

    vector<map_point *> vpCheckFuseMapPoint; // map_point vector from current map to allow to fuse duplicated points with the old map (merge)
    vector<key_frame *> vpCurrentConnectedKFs;

    merge_connected_key_frames.push_back(merge_matched_key_frame);
    auto aux = merge_matched_key_frame->get_vector_covisible_key_frames();
    merge_connected_key_frames.insert(merge_connected_key_frames.end(), aux.begin(), aux.end());
    if (merge_connected_key_frames.size() > 6)
    {
        merge_connected_key_frames.erase(merge_connected_key_frames.begin() + 6, merge_connected_key_frames.end());
    }

    current_key_frame->update_connections();
    vpCurrentConnectedKFs.push_back(current_key_frame);
    aux = current_key_frame->get_vector_covisible_key_frames();
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end());
    if (vpCurrentConnectedKFs.size() > 6)
    {
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin() + 6, vpCurrentConnectedKFs.end());
    }

    set<map_point *> spMapPointMerge;
    for (auto pKFi : merge_connected_key_frames)
    {
        auto vpMPs = pKFi->get_map_points();
        spMapPointMerge.insert(vpMPs.begin(), vpMPs.end());
        if (spMapPointMerge.size() > 1000)
        {
            break;
        }
    }

    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    search_and_fuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint);

    for (auto pKFi : vpCurrentConnectedKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        pKFi->update_connections();
    }
    for (auto pKFi : merge_connected_key_frames)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        pKFi->update_connections();
    }

    // TODO Check: If new map is too small, we suppose that not informaiton can be propagated from new to old map
    if (numKFnew < 10)
    {
        local_mapper->release();
        return;
    }

    // Perform BA
    auto bStopFlag = false;
    auto pCurrKF   = tracker->get_last_key_frame();
    optimizer::merge_inertial_ba(_cameras, pCurrKF, merge_matched_key_frame, &bStopFlag, pCurrentMap, CorrectedSim3);

    // Release Local Mapping.
    local_mapper->release();

    return;
}

void loop_closing::check_observations(const set<key_frame *> &spKFsMap1, const set<key_frame *> &spKFsMap2)
{
    spdlog::debug("----------------------");
    for (const auto pKFi1 : spKFsMap1)
    {
        std::map<key_frame *, int> mMatchedMP;
        auto spMPs = pKFi1->get_map_points();

        for (const auto pMPij : spMPs)
        {
            if (!pMPij || pMPij->is_bad())
            {
                continue;
            }

            auto mMPijObs = pMPij->get_observations();
            for (auto pKFi2 : spKFsMap2)
            {
                if (mMPijObs.find(pKFi2) != mMPijObs.end())
                {
                    if (mMatchedMP.find(pKFi2) != mMatchedMP.end())
                    {
                        mMatchedMP[pKFi2] = mMatchedMP[pKFi2] + 1;
                    }
                    else
                    {
                        mMatchedMP[pKFi2] = 1;
                    }
                }
            }
        }

        if (mMatchedMP.size() == 0)
        {
            spdlog::debug("CHECK-OBS: keyframe {} has no matched map points in the other map", pKFi1->id);
        }
        else
        {
            spdlog::debug("CHECK-OBS: keyframe {} has matched map points with {} keyframes in the other map", pKFi1->id, mMatchedMP.size());
            for (pair<key_frame *, int> matchedKF : mMatchedMP)
            {
                spdlog::trace("Keyframe {} has {} matches", matchedKF.first->id, matchedKF.second);
            }
        }
    }
    spdlog::debug("----------------------");
}

void loop_closing::search_and_fuse(const key_frame_and_pose &CorrectedPosesMap, const vector<map_point *> &vpMapPoints) const
{
    descriptor_matcher matcher(0.8);

    auto total_replaces = 0;

    for (auto mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++)
    {
        auto num_replaces = 0;
        const auto pKFi         = mit->first;
        const auto pMap         = pKFi->get_map();

        auto g2oScw = mit->second;
        auto Scw    = converter::to_sophus(g2oScw);

        vector<map_point *>  vpReplacePoints(vpMapPoints.size(), static_cast<map_point *>(NULL));
        [[maybe_unused]] auto numFused = matcher.fuse(_cameras, pKFi, Scw, vpMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->map_update_mutex);
        const int          nLP = vpMapPoints.size();
        for (auto i = 0; i < nLP; i++)
        {
            const auto pRep = vpReplacePoints[i];
            if (pRep)
            {

                num_replaces += 1;
                pRep->replace(vpMapPoints[i]);
            }
        }

        total_replaces += num_replaces;
    }
}

void loop_closing::search_and_fuse(const vector<key_frame *> &vConectedKFs, const vector<map_point *> &vpMapPoints) const
{
    descriptor_matcher matcher(0.8);

    [[maybe_unused]] auto total_replaces = 0;

    for (auto mit = vConectedKFs.begin(), mend = vConectedKFs.end(); mit != mend; mit++)
    {
        auto num_replaces = 0;
        const auto pKF          = *mit;
        const auto pMap         = pKF->get_map();
        auto Tcw          = pKF->get_pose();
        Sophus::Sim3f  Scw(Tcw.unit_quaternion(), Tcw.translation());
        Scw.setScale(1.f);
        vector<map_point *> vpReplacePoints(vpMapPoints.size(), static_cast<map_point *>(NULL));
        matcher.fuse(_cameras, pKF, Scw, vpMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->map_update_mutex);
        const int          nLP = vpMapPoints.size();
        for (auto i = 0; i < nLP; i++)
        {
            const auto pRep = vpReplacePoints[i];
            if (pRep)
            {
                num_replaces += 1;
                pRep->replace(vpMapPoints[i]);
            }
        }
    }
}

void loop_closing::request_reset()
{
    {
        unique_lock<mutex> lock(reset_mutex);
        reset_requested = true;
    }

    while (1)
    {
        {
            unique_lock<mutex> lock2(reset_mutex);
            if (!reset_requested)
            {
                break;
            }
        }
        usleep(5000);
    }
}

void loop_closing::request_reset_active_map(map *pMap)
{
    {
        unique_lock<mutex> lock(reset_mutex);
        active_map_reset_requested = true;
        map_to_reset               = pMap;
    }

    while (1)
    {
        {
            unique_lock<mutex> lock2(reset_mutex);
            if (!active_map_reset_requested)
            {
                break;
            }
        }
        usleep(3000);
    }
}

void loop_closing::reset_if_requested()
{
    unique_lock<mutex> lock(reset_mutex);
    if (reset_requested)
    {
        spdlog::debug("Loop closer reset requested");
        loop_key_frame_queue.clear();
        last_loop_key_frame_id     = 0; // TODO old variable, it is not use in the new algorithm
        reset_requested            = false;
        active_map_reset_requested = false;
    }
    else if (active_map_reset_requested)
    {

        for (list<key_frame *>::const_iterator it = loop_key_frame_queue.begin(); it != loop_key_frame_queue.end();)
        {
            const auto pKFi = *it;
            if (pKFi->get_map() == map_to_reset)
            {
                it = loop_key_frame_queue.erase(it);
            }
            else
            {
                ++it;
            }
        }

        last_loop_key_frame_id     = owner_atlas->get_last_init_kf_id(); // TODO old variable, it is not use in the new algorithm
        active_map_reset_requested = false;
    }
}

void loop_closing::run_global_bundle_adjustment(map *pActiveMap, unsigned long nLoopKF)
{
    const processing_statistics::backend_timer statistics_timer(_statistics, processing_statistics::backend_stage::global_bundle_adjustment);
    spdlog::debug("Starting Global Bundle Adjustment");

    const auto bImuInit = pActiveMap->is_imu_initialized();

    if (!bImuInit)
    {
        optimizer::global_bundle_adjustment(_cameras, pActiveMap, 10, &global_bundle_adjustment_stop_requested, nLoopKF, false);
    }
    else
    {
        optimizer::full_inertial_ba(_cameras, pActiveMap, 7, false, nLoopKF, &global_bundle_adjustment_stop_requested);
    }

    const auto idx = full_bundle_adjustment_index;

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(global_bundle_adjustment_mutex);
        if (idx != full_bundle_adjustment_index)
        {
            return;
        }

        if (!bImuInit && pActiveMap->is_imu_initialized())
        {
            return;
        }

        if (!global_bundle_adjustment_stop_requested)
        {
            spdlog::debug("Global Bundle Adjustment finished");
            spdlog::debug("Updating map ...");

            local_mapper->request_stop();
            // Wait until Local Mapping has effectively stopped

            while (!local_mapper->is_stopped() && !local_mapper->is_finished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->map_update_mutex);

            //  Correct keyframes starting at map first keyframe
            list<key_frame *> lpKFtoCheck(pActiveMap->key_frame_origins.begin(), pActiveMap->key_frame_origins.end());

            while (!lpKFtoCheck.empty())
            {
                const auto pKF     = lpKFtoCheck.front();
                const auto sChilds = pKF->get_children();
                auto Twc     = pKF->get_pose_inverse();
                for (auto sit = sChilds.begin(); sit != sChilds.end(); sit++)
                {
                    auto pChild = *sit;
                    if (!pChild || pChild->is_bad())
                    {
                        continue;
                    }

                    if (pChild->global_ba_key_frame_id != nLoopKF)
                    {
                        auto Tchildc          = pChild->get_pose() * Twc;
                        pChild->camera_pose_global_ba = Tchildc * pKF->camera_pose_global_ba; //*Tcorc*pKF->camera_pose_global_ba;

                        auto Rcor = pChild->camera_pose_global_ba.so3().inverse() * pChild->get_pose().so3();
                        if (pChild->is_velocity_set())
                        {
                            pChild->velocity_world_global_ba = Rcor * pChild->get_velocity();
                        }
                        else
                        {
                            spdlog::debug("Child velocity empty!! ");
                        }

                        pChild->bias_global_ba = pChild->get_imu_bias();

                        pChild->global_ba_key_frame_id = nLoopKF;
                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->camera_pose_before_global_ba = pKF->get_pose();
                pKF->set_pose(pKF->camera_pose_global_ba);

                if (pKF->has_imu)
                {
                    pKF->velocity_world_before_global_ba = pKF->get_velocity();

                    pKF->set_velocity(pKF->velocity_world_global_ba);
                    pKF->set_new_bias(pKF->bias_global_ba);
                }

                lpKFtoCheck.pop_front();
            }

            //  Correct MapPoints
            const auto vpMPs = pActiveMap->get_all_map_points();

            for (size_t i = 0; std::cmp_less(i, vpMPs.size()); i++)
            {
                const auto pMP = vpMPs[i];

                if (pMP->is_bad())
                {
                    continue;
                }

                if (pMP->global_ba_key_frame_id == nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->set_world_position(pMP->global_ba_position);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    const auto pRefKF = pMP->get_reference_key_frame();

                    if (pRefKF->global_ba_key_frame_id != nLoopKF)
                    {
                        continue;
                    }

                    // Map to non-corrected camera
                    // cv::Mat Rcw = pRefKF->camera_pose_before_global_ba.rowRange(0,3).colRange(0,3);
                    // cv::Mat tcw = pRefKF->camera_pose_before_global_ba.rowRange(0,3).col(3);
                    auto Xc = pRefKF->camera_pose_before_global_ba * pMP->get_world_position();

                    // Backproject using corrected camera
                    pMP->set_world_position(pRefKF->get_pose_inverse() * Xc);
                }
            }

            pActiveMap->inform_new_big_change();
            pActiveMap->increase_change_index();

            // TODO Check this update

            local_mapper->release();

            spdlog::debug("Map updated!");
        }

        global_bundle_adjustment_finished = true;
        global_bundle_adjustment_running  = false;
    }
}

void loop_closing::stop_global_bundle_adjustment()
{
    {
        const std::lock_guard lock(global_bundle_adjustment_mutex);
        global_bundle_adjustment_stop_requested = true;
        ++full_bundle_adjustment_index;
    }
    if (_global_ba_thread.joinable())
    {
        _global_ba_thread.request_stop();
        _global_ba_thread.join();
    }
    const std::lock_guard lock(global_bundle_adjustment_mutex);
    global_bundle_adjustment_running  = false;
    global_bundle_adjustment_finished = true;
}

void loop_closing::request_finish()
{
    unique_lock<mutex> lock(finish_mutex);
    finish_requested = true;
}

bool loop_closing::check_finish()
{
    unique_lock<mutex> lock(finish_mutex);
    return finish_requested;
}

void loop_closing::set_finish()
{
    unique_lock<mutex> lock(finish_mutex);
    finished = true;
}

bool loop_closing::is_finished()
{
    unique_lock<mutex> lock(finish_mutex);
    return finished;
}

} // namespace ORB_SLAM3
