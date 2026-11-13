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

#include "local_mapping.h"

#include "converter.h"
#include "descriptor_matcher.h"
#include "geometric_tools.h"
#include "loop_closing.h"
#include "optimizer.h"

#include <mutex>

#include <gsl/gsl>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

local_mapping::local_mapping(processing_statistics &statistics, system *pSys, atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName)
    : scale(1.0), initialization_section(0), initialization_index(0), iteration_index(0), skip_bundle_adjustment_1(true), skip_bundle_adjustment_2(true), _statistics(statistics),
      owner_system(pSys), monocular(bMonocular), inertial(bInertial), reset_requested(false), active_map_reset_requested(false), finish_requested(false), finished(true), owner_atlas(pAtlas),
      bundle_adjustment_aborted(false), stopped(false), stop_requested_flag(false), not_stop_requested(false), accept_key_frames_flag(true), initializing(false),
      inertial_information(Eigen::MatrixXd::Zero(9, 9))
{
    inlier_match_count = 0;

    bad_imu = false;

    initialization_elapsed_time = 0.f;

    local_map_point_count   = 0;
    key_frame_culling_count = 0;
}

void local_mapping::set_loop_closer(loop_closing *pLoopCloser) { loop_closer = pLoopCloser; }

void local_mapping::set_tracker(tracking *pTracker) { tracker = pTracker; }

void local_mapping::run()
{
    finished = false;

    while (1)
    {
        // Tracking will see that Local Mapping is busy
        set_accept_key_frames(false);

        // Check if there are keyframes in the queue
        if (check_new_key_frames() && !bad_imu)
        {
            const processing_statistics::backend_timer statistics_timer(_statistics, processing_statistics::backend_stage::local_mapping);
            // BoW conversion and insertion in Map
            process_new_key_frame();

            // Check recent MapPoints
            cull_map_points();

            // Triangulate new MapPoints
            create_new_map_points();

            bundle_adjustment_aborted = false;

            if (!check_new_key_frames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                search_in_neighbors();
            }

            [[maybe_unused]] bool b_doneLBA      = false;
            int                   num_FixedKF_BA = 0;
            int                   num_OptKF_BA   = 0;
            int                   num_MPs_BA     = 0;
            int                   num_edges_BA   = 0;

            if (!check_new_key_frames() && !stop_requested())
            {
                if (owner_atlas->key_frames_in_map() > 2)
                {

                    if (inertial && current_key_frame->get_map()->is_imu_initialized())
                    {
                        float dist = (current_key_frame->previous_key_frame->get_camera_center() - current_key_frame->get_camera_center()).norm() +
                                     (current_key_frame->previous_key_frame->previous_key_frame->get_camera_center() - current_key_frame->previous_key_frame->get_camera_center()).norm();

                        if (dist > 0.05)
                        {
                            initialization_elapsed_time += current_key_frame->timestamp - current_key_frame->previous_key_frame->timestamp;
                        }
                        if (!current_key_frame->get_map()->get_inertial_ba2())
                        {
                            if ((initialization_elapsed_time < 10.f) && (dist < 0.02))
                            {
                                spdlog::warn("Not enough motion for initialization; resetting active map");
                                unique_lock<mutex> lock(reset_mutex);
                                active_map_reset_requested = true;
                                map_to_reset               = current_key_frame->get_map();
                                bad_imu                    = true;
                            }
                        }

                        bool bLarge = ((tracker->get_matches_inliers() > 75) && monocular) || ((tracker->get_matches_inliers() > 100) && !monocular);
                        optimizer::local_inertial_ba(current_key_frame,
                                                     &bundle_adjustment_aborted,
                                                     current_key_frame->get_map(),
                                                     num_FixedKF_BA,
                                                     num_OptKF_BA,
                                                     num_MPs_BA,
                                                     num_edges_BA,
                                                     bLarge,
                                                     !current_key_frame->get_map()->get_inertial_ba2());
                        b_doneLBA = true;
                    }
                    else
                    {
                        optimizer::local_bundle_adjustment(current_key_frame, &bundle_adjustment_aborted, current_key_frame->get_map(), num_FixedKF_BA, num_OptKF_BA, num_MPs_BA, num_edges_BA);
                        b_doneLBA = true;
                    }
                }

                // Initialize IMU here
                if (!current_key_frame->get_map()->is_imu_initialized() && inertial)
                {
                    if (monocular)
                    {
                        initialize_imu(1e2, 1e10, true);
                    }
                    else
                    {
                        initialize_imu(1e2, 1e5, true);
                    }
                }

                // Check redundant local Keyframes
                cull_key_frames();

                if ((initialization_elapsed_time < 50.0f) && inertial)
                {
                    if (current_key_frame->get_map()->is_imu_initialized() && tracker->state == tracking::OK) // Enter here everytime local-mapping is called
                    {
                        if (!current_key_frame->get_map()->get_inertial_ba1())
                        {
                            if (initialization_elapsed_time > 5.0f)
                            {
                                spdlog::info("Starting VIBA stage 1");
                                current_key_frame->get_map()->set_inertial_ba1();
                                if (monocular)
                                {
                                    initialize_imu(1.f, 1e5, true);
                                }
                                else
                                {
                                    initialize_imu(1.f, 1e5, true);
                                }

                                spdlog::info("Finished VIBA stage 1");
                            }
                        }
                        else if (!current_key_frame->get_map()->get_inertial_ba2())
                        {
                            if (initialization_elapsed_time > 15.0f)
                            {
                                spdlog::info("Starting VIBA stage 2");
                                current_key_frame->get_map()->set_inertial_ba2();
                                if (monocular)
                                {
                                    initialize_imu(0.f, 0.f, true);
                                }
                                else
                                {
                                    initialize_imu(0.f, 0.f, true);
                                }

                                spdlog::info("Finished VIBA stage 2");
                            }
                        }

                        // scale refinement
                        if (((owner_atlas->key_frames_in_map()) <= 200) &&
                            ((initialization_elapsed_time > 25.0f && initialization_elapsed_time < 25.5f) || (initialization_elapsed_time > 35.0f && initialization_elapsed_time < 35.5f) ||
                             (initialization_elapsed_time > 45.0f && initialization_elapsed_time < 45.5f) || (initialization_elapsed_time > 55.0f && initialization_elapsed_time < 55.5f) ||
                             (initialization_elapsed_time > 65.0f && initialization_elapsed_time < 65.5f) || (initialization_elapsed_time > 75.0f && initialization_elapsed_time < 75.5f)))
                        {
                            if (monocular)
                            {
                                refine_scale();
                            }
                        }
                    }
                }
            }

            loop_closer->insert_key_frame(current_key_frame);
        }
        else if (stop() && !bad_imu)
        {
            // Safe area to stop
            while (is_stopped() && !check_finish())
            {
                usleep(3000);
            }
            if (check_finish())
            {
                break;
            }
        }

        reset_if_requested();

        // Tracking will see that Local Mapping is busy
        set_accept_key_frames(true);

        if (check_finish())
        {
            break;
        }

        usleep(3000);
    }

    set_finish();
}

void local_mapping::insert_key_frame(key_frame *pKF)
{
    unique_lock<mutex> lock(new_key_frames_mutex);
    new_key_frames.push_back(pKF);
    bundle_adjustment_aborted = true;
}

bool local_mapping::check_new_key_frames()
{
    unique_lock<mutex> lock(new_key_frames_mutex);
    return (!new_key_frames.empty());
}

void local_mapping::process_new_key_frame()
{
    {
        unique_lock<mutex> lock(new_key_frames_mutex);
        current_key_frame = new_key_frames.front();
        new_key_frames.pop_front();
    }

    // Compute Bags of Words structures
    current_key_frame->compute_bow();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<map_point *> vpMapPointMatches = current_key_frame->get_map_point_matches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        map_point *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->is_bad())
            {
                if (!pMP->is_in_key_frame(current_key_frame))
                {
                    pMP->add_observation(current_key_frame, i);
                    pMP->update_normal_and_depth();
                    pMP->compute_distinctive_descriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    recently_added_map_points.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    current_key_frame->update_connections();

    // Insert Keyframe in Map
    owner_atlas->add_key_frame(current_key_frame);
}

void local_mapping::empty_queue()
{
    while (check_new_key_frames())
    {
        process_new_key_frame();
    }
}

void local_mapping::cull_map_points()
{
    // Check Recent Added MapPoints
    list<map_point *>::iterator lit          = recently_added_map_points.begin();
    const unsigned long int     nCurrentKFid = current_key_frame->id;

    int nThObs;
    if (monocular)
    {
        nThObs = 2;
    }
    else
    {
        nThObs = 3;
    }
    const int cnThObs = nThObs;

    int borrar = recently_added_map_points.size();

    while (lit != recently_added_map_points.end())
    {
        map_point *pMP = *lit;

        if (pMP->is_bad())
        {
            lit = recently_added_map_points.erase(lit);
        }
        else if (pMP->get_found_ratio() < 0.25f)
        {
            pMP->set_bad_flag();
            lit = recently_added_map_points.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->first_key_frame_id) >= 2 && pMP->get_observation_count() <= cnThObs)
        {
            pMP->set_bad_flag();
            lit = recently_added_map_points.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->first_key_frame_id) >= 3)
        {
            lit = recently_added_map_points.erase(lit);
        }
        else
        {
            lit++;
            borrar--;
        }
    }
}

void local_mapping::create_new_map_points()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // For stereo inertial case
    if (monocular)
    {
        nn = 30;
    }
    vector<key_frame *> vpNeighKFs = current_key_frame->get_best_covisibility_key_frames(nn);

    if (inertial)
    {
        key_frame *pKF   = current_key_frame;
        int        count = 0;
        while ((vpNeighKFs.size() <= gsl::narrow_cast<size_t>(nn)) && (pKF->previous_key_frame) && (count++ < nn))
        {
            vector<key_frame *>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->previous_key_frame);
            if (it == vpNeighKFs.end())
            {
                vpNeighKFs.push_back(pKF->previous_key_frame);
            }
            pKF = pKF->previous_key_frame;
        }
    }

    float th = 0.6f;

    descriptor_matcher matcher(th, false);

    Sophus::SE3<float>         sophTcw1 = current_key_frame->get_pose();
    Eigen::Matrix<float, 3, 4> eigTcw1  = sophTcw1.matrix3x4();
    Eigen::Matrix<float, 3, 3> Rcw1     = eigTcw1.block<3, 3>(0, 0);
    Eigen::Matrix<float, 3, 3> Rwc1     = Rcw1.transpose();
    Eigen::Vector3f            tcw1     = sophTcw1.translation();
    Eigen::Vector3f            Ow1      = current_key_frame->get_camera_center();

    const float                  &fx1    = current_key_frame->fx;
    const float                  &fy1    = current_key_frame->fy;
    const float                  &cx1    = current_key_frame->cx;
    const float                  &cy1    = current_key_frame->cy;
    [[maybe_unused]] const float &invfx1 = current_key_frame->invfx;
    [[maybe_unused]] const float &invfy1 = current_key_frame->invfy;

    const float ratioFactor         = 1.5f * current_key_frame->scale_factor;
    int         countStereo         = 0;
    int         countStereoGoodProj = 0;
    int         countStereoAttempt  = 0;
    int         totalStereoPts      = 0;
    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        if (i > 0 && check_new_key_frames())
        {
            return;
        }

        key_frame *pKF2 = vpNeighKFs[i];

        size_t camera_index1 = 0;
        size_t camera_index2 = 0;

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2       = pKF2->get_camera_center();
        Eigen::Vector3f vBaseline = Ow2 - Ow1;
        const float     baseline  = vBaseline.norm();

        if (!monocular)
        {
            if (baseline < pKF2->mb)
            {
                continue;
            }
        }
        else
        {
            const float medianDepthKF2     = pKF2->compute_scene_median_depth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
            {
                continue;
            }
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t, size_t>> vMatchedIndices;
        bool                         bCoarse = inertial && tracker->state == tracking::RECENTLY_LOST && current_key_frame->get_map()->get_inertial_ba2();

        matcher.search_for_triangulation(current_key_frame, pKF2, vMatchedIndices, false, bCoarse);

        Sophus::SE3<float>         sophTcw2 = pKF2->get_pose();
        Eigen::Matrix<float, 3, 4> eigTcw2  = sophTcw2.matrix3x4();
        Eigen::Matrix<float, 3, 3> Rcw2     = eigTcw2.block<3, 3>(0, 0);
        Eigen::Matrix<float, 3, 3> Rwc2     = Rcw2.transpose();
        Eigen::Vector3f            tcw2     = sophTcw2.translation();

        const float                  &fx2    = pKF2->fx;
        const float                  &fy2    = pKF2->fy;
        const float                  &cx2    = pKF2->cx;
        const float                  &cy2    = pKF2->cy;
        [[maybe_unused]] const float &invfx2 = pKF2->invfx;
        [[maybe_unused]] const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for (int ikp = 0; ikp < nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const auto &kp1      = ((!is_fisheye_stereo(current_key_frame->cameras))             ? current_key_frame->keypoints[idx1]
                                    : (std::cmp_less(idx1, current_key_frame->keypoints.size())) ? current_key_frame->keypoints[idx1]
                                                                                                 : current_key_frame->right_keypoints[idx1 - current_key_frame->keypoints.size()]);
            const float kp1_ur   = current_key_frame->right_coordinates[idx1];
            bool        bStereo1 = (!is_fisheye_stereo(current_key_frame->cameras) && kp1_ur >= 0);
            const bool  bRight1  = is_fisheye_stereo(current_key_frame->cameras) && std::cmp_greater_equal(idx1, current_key_frame->keypoints.size());

            const auto &kp2 = ((!is_fisheye_stereo(pKF2->cameras))           ? pKF2->keypoints[idx2]
                               : std::cmp_less(idx2, pKF2->keypoints.size()) ? pKF2->keypoints[idx2]
                                                                             : pKF2->right_keypoints[idx2 - pKF2->keypoints.size()]);

            const float kp2_ur   = pKF2->right_coordinates[idx2];
            bool        bStereo2 = (!is_fisheye_stereo(pKF2->cameras) && kp2_ur >= 0);
            const bool  bRight2  = is_fisheye_stereo(pKF2->cameras) && std::cmp_greater_equal(idx2, pKF2->keypoints.size());

            if (is_fisheye_stereo(current_key_frame->cameras) && is_fisheye_stereo(pKF2->cameras))
            {
                if (bRight1 && bRight2)
                {
                    sophTcw1 = current_key_frame->get_right_pose();
                    Ow1      = current_key_frame->get_right_camera_center();

                    sophTcw2 = pKF2->get_right_pose();
                    Ow2      = pKF2->get_right_camera_center();

                    camera_index1 = 1;
                    camera_index2 = 1;
                }
                else if (bRight1 && !bRight2)
                {
                    sophTcw1 = current_key_frame->get_right_pose();
                    Ow1      = current_key_frame->get_right_camera_center();

                    sophTcw2 = pKF2->get_pose();
                    Ow2      = pKF2->get_camera_center();

                    camera_index1 = 1;
                    camera_index2 = 0;
                }
                else if (!bRight1 && bRight2)
                {
                    sophTcw1 = current_key_frame->get_pose();
                    Ow1      = current_key_frame->get_camera_center();

                    sophTcw2 = pKF2->get_right_pose();
                    Ow2      = pKF2->get_right_camera_center();

                    camera_index1 = 0;
                    camera_index2 = 1;
                }
                else
                {
                    sophTcw1 = current_key_frame->get_pose();
                    Ow1      = current_key_frame->get_camera_center();

                    sophTcw2 = pKF2->get_pose();
                    Ow2      = pKF2->get_camera_center();

                    camera_index1 = 0;
                    camera_index2 = 0;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1    = eigTcw1.block<3, 3>(0, 0);
                Rwc1    = Rcw1.transpose();
                tcw1    = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2    = eigTcw2.block<3, 3>(0, 0);
                Rwc2    = Rcw2.transpose();
                tcw2    = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = current_key_frame->cameras[camera_index1]->unproject_eigen(kp1.pt_undistorted);
            Eigen::Vector3f xn2 = pKF2->cameras[camera_index2]->unproject_eigen(kp2.pt_undistorted);

            Eigen::Vector3f ray1            = Rwc1 * xn1;
            Eigen::Vector3f ray2            = Rwc2 * xn2;
            const float     cosParallaxRays = ray1.dot(ray2) / (ray1.norm() * ray2.norm());

            float cosParallaxStereo  = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1)
            {
                cosParallaxStereo1 = cos(2 * atan2(current_key_frame->mb / 2, current_key_frame->depths[idx1]));
            }
            else if (bStereo2)
            {
                cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->depths[idx2]));
            }

            if (bStereo1 || bStereo2)
            {
                totalStereoPts++;
            }

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

            Eigen::Vector3f x3D;

            bool goodProj     = false;
            bool bPointStereo = false;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 || (cosParallaxRays < 0.9996 && inertial) || (cosParallaxRays < 0.9998 && !inertial)))
            {
                goodProj = geometric_tools::triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if (!goodProj)
                {
                    continue;
                }
            }
            else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj     = current_key_frame->unproject_stereo(idx1, x3D);
            }
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj     = pKF2->unproject_stereo(idx2, x3D);
            }
            else
            {
                continue; // No stereo and very low parallax
            }

            if (goodProj && bPointStereo)
            {
                countStereoGoodProj++;
            }

            if (!goodProj)
            {
                continue;
            }

            // Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if (z1 <= 0)
            {
                continue;
            }

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if (z2 <= 0)
            {
                continue;
            }

            // Check reprojection error in first keyframe
            const float &sigmaSquare1 = current_key_frame->level_sigma_squared[kp1.octave];
            const float  x1           = Rcw1.row(0).dot(x3D) + tcw1(0);
            const float  y1           = Rcw1.row(1).dot(x3D) + tcw1(1);
            const float  invz1        = 1.0 / z1;

            if (!bStereo1)
            {
                cv::Point2f uv1   = current_key_frame->cameras[camera_index1]->project(cv::Point3f(x1, y1, z1));
                float       errX1 = uv1.x - kp1.pt_undistorted.x;
                float       errY1 = uv1.y - kp1.pt_undistorted.y;

                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                {
                    continue;
                }
            }
            else
            {
                float u1      = fx1 * x1 * invz1 + cx1;
                float u1_r    = u1 - current_key_frame->mbf * invz1;
                float v1      = fy1 * y1 * invz1 + cy1;
                float errX1   = u1 - kp1.pt_undistorted.x;
                float errY1   = v1 - kp1.pt_undistorted.y;
                float errX1_r = u1_r - kp1_ur;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                {
                    continue;
                }
            }

            // Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->level_sigma_squared[kp2.octave];
            const float x2           = Rcw2.row(0).dot(x3D) + tcw2(0);
            const float y2           = Rcw2.row(1).dot(x3D) + tcw2(1);
            const float invz2        = 1.0 / z2;
            if (!bStereo2)
            {
                cv::Point2f uv2   = pKF2->cameras[camera_index2]->project(cv::Point3f(x2, y2, z2));
                float       errX2 = uv2.x - kp2.pt_undistorted.x;
                float       errY2 = uv2.y - kp2.pt_undistorted.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                {
                    continue;
                }
            }
            else
            {
                float u2      = fx2 * x2 * invz2 + cx2;
                float u2_r    = u2 - current_key_frame->mbf * invz2;
                float v2      = fy2 * y2 * invz2 + cy2;
                float errX2   = u2 - kp2.pt_undistorted.x;
                float errY2   = v2 - kp2.pt_undistorted.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                {
                    continue;
                }
            }

            // Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1;
            float           dist1   = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float           dist2   = normal2.norm();

            if (dist1 == 0 || dist2 == 0)
            {
                continue;
            }

            if (far_points_enabled && (dist1 >= far_point_threshold || dist2 >= far_point_threshold)) // MODIFICATION
            {
                continue;
            }

            const float ratioDist   = dist2 / dist1;
            const float ratioOctave = current_key_frame->scale_factors[kp1.octave] / pKF2->scale_factors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
            {
                continue;
            }

            // Triangulation is succesfull
            map_point *pMP = new map_point(x3D, current_key_frame, owner_atlas->get_current_map());
            if (bPointStereo)
            {
                countStereo++;
            }

            pMP->add_observation(current_key_frame, idx1);
            pMP->add_observation(pKF2, idx2);

            current_key_frame->add_map_point(pMP, idx1);
            pKF2->add_map_point(pMP, idx2);

            pMP->compute_distinctive_descriptors();

            pMP->update_normal_and_depth();

            owner_atlas->add_map_point(pMP);
            recently_added_map_points.push_back(pMP);
        }
    }
}

void local_mapping::search_in_neighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if (monocular)
    {
        nn = 30;
    }
    const vector<key_frame *> vpNeighKFs = current_key_frame->get_best_covisibility_key_frames(nn);
    vector<key_frame *>       vpTargetKFs;
    for (vector<key_frame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        key_frame *pKFi = *vit;
        if (pKFi->is_bad() || pKFi->fuse_target_key_frame_id == current_key_frame->id)
        {
            continue;
        }
        vpTargetKFs.push_back(pKFi);
        pKFi->fuse_target_key_frame_id = current_key_frame->id;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for (int i = 0, imax = vpTargetKFs.size(); i < imax; i++)
    {
        const vector<key_frame *> vpSecondNeighKFs = vpTargetKFs[i]->get_best_covisibility_key_frames(20);
        for (vector<key_frame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            key_frame *pKFi2 = *vit2;
            if (pKFi2->is_bad() || pKFi2->fuse_target_key_frame_id == current_key_frame->id || pKFi2->id == current_key_frame->id)
            {
                continue;
            }
            vpTargetKFs.push_back(pKFi2);
            pKFi2->fuse_target_key_frame_id = current_key_frame->id;
        }
        if (bundle_adjustment_aborted)
        {
            break;
        }
    }

    // Extend to temporal neighbors
    if (inertial)
    {
        key_frame *pKFi = current_key_frame->previous_key_frame;
        while (vpTargetKFs.size() < 20 && pKFi)
        {
            if (pKFi->is_bad() || pKFi->fuse_target_key_frame_id == current_key_frame->id)
            {
                pKFi = pKFi->previous_key_frame;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->fuse_target_key_frame_id = current_key_frame->id;
            pKFi                           = pKFi->previous_key_frame;
        }
    }

    // Search matches by projection from current KF in target KFs
    descriptor_matcher  matcher;
    vector<map_point *> vpMapPointMatches = current_key_frame->get_map_point_matches();
    for (vector<key_frame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
    {
        key_frame *pKFi = *vit;

        matcher.fuse(pKFi, vpMapPointMatches);
        if (is_fisheye_stereo(pKFi->cameras))
        {
            matcher.fuse(pKFi, vpMapPointMatches, true);
        }
    }

    if (bundle_adjustment_aborted)
    {
        return;
    }

    // Search matches by projection from target KFs in current KF
    vector<map_point *> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (vector<key_frame *>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
    {
        key_frame *pKFi = *vitKF;

        vector<map_point *> vpMapPointsKFi = pKFi->get_map_point_matches();

        for (vector<map_point *>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
        {
            map_point *pMP = *vitMP;
            if (!pMP)
            {
                continue;
            }
            if (pMP->is_bad() || pMP->fuse_candidate_key_frame_id == current_key_frame->id)
            {
                continue;
            }
            pMP->fuse_candidate_key_frame_id = current_key_frame->id;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.fuse(current_key_frame, vpFuseCandidates);
    if (is_fisheye_stereo(current_key_frame->cameras))
    {
        matcher.fuse(current_key_frame, vpFuseCandidates, true);
    }

    // Update points
    vpMapPointMatches = current_key_frame->get_map_point_matches();
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
    {
        map_point *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->is_bad())
            {
                pMP->compute_distinctive_descriptors();
                pMP->update_normal_and_depth();
            }
        }
    }

    // Update connections in covisibility graph
    current_key_frame->update_connections();
}

void local_mapping::request_stop()
{
    unique_lock<mutex> lock(stop_mutex);
    stop_requested_flag = true;
    unique_lock<mutex> lock2(new_key_frames_mutex);
    bundle_adjustment_aborted = true;
}

bool local_mapping::stop()
{
    unique_lock<mutex> lock(stop_mutex);
    if (stop_requested_flag && !not_stop_requested)
    {
        stopped = true;
        spdlog::debug("Local mapping stopped");
        return true;
    }

    return false;
}

bool local_mapping::is_stopped()
{
    unique_lock<mutex> lock(stop_mutex);
    return stopped;
}

bool local_mapping::stop_requested()
{
    unique_lock<mutex> lock(stop_mutex);
    return stop_requested_flag;
}

void local_mapping::release()
{
    unique_lock<mutex> lock(stop_mutex);
    unique_lock<mutex> lock2(finish_mutex);
    if (finished)
    {
        return;
    }
    stopped             = false;
    stop_requested_flag = false;
    for (list<key_frame *>::iterator lit = new_key_frames.begin(), lend = new_key_frames.end(); lit != lend; lit++)
    {
        delete *lit;
    }
    new_key_frames.clear();

    spdlog::debug("Local mapping released");
}

bool local_mapping::accept_key_frames()
{
    unique_lock<mutex> lock(accept_key_frames_mutex);
    return accept_key_frames_flag;
}

void local_mapping::set_accept_key_frames(bool flag)
{
    unique_lock<mutex> lock(accept_key_frames_mutex);
    accept_key_frames_flag = flag;
}

bool local_mapping::set_not_stop(bool flag)
{
    unique_lock<mutex> lock(stop_mutex);

    if (flag && stopped)
    {
        return false;
    }

    not_stop_requested = flag;

    return true;
}

void local_mapping::interrupt_bundle_adjustment() { bundle_adjustment_aborted = true; }

void local_mapping::cull_key_frames()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    current_key_frame->update_best_covisibles();
    vector<key_frame *> vpLocalKeyFrames = current_key_frame->get_vector_covisible_key_frames();

    float redundant_th;
    if (!inertial)
    {
        redundant_th = 0.9;
    }
    else if (monocular)
    {
        redundant_th = 0.9;
    }
    else
    {
        redundant_th = 0.5;
    }

    const bool bInitImu = owner_atlas->is_imu_initialized();
    int        count    = 0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID = 0;
    if (inertial)
    {
        int        count  = 0;
        key_frame *aux_KF = current_key_frame;
        while (count < Nd && aux_KF->previous_key_frame)
        {
            aux_KF = aux_KF->previous_key_frame;
            count++;
        }
        last_ID = aux_KF->id;
    }

    for (vector<key_frame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
    {
        count++;
        key_frame *pKF = *vit;

        if ((pKF->id == pKF->get_map()->get_initial_key_frame_id()) || pKF->is_bad())
        {
            continue;
        }
        const vector<map_point *> vpMapPoints = pKF->get_map_point_matches();

        int       nObs                   = 3;
        const int thObs                  = nObs;
        int       nRedundantObservations = 0;
        int       nMPs                   = 0;
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            map_point *pMP = vpMapPoints[i];
            if (pMP)
            {
                if (!pMP->is_bad())
                {
                    if (!monocular)
                    {
                        if (pKF->depths[i] > pKF->depth_threshold || pKF->depths[i] < 0)
                        {
                            continue;
                        }
                    }

                    nMPs++;
                    if (pMP->get_observation_count() > thObs)
                    {
                        const int                                   &scaleLevel   = (!is_fisheye_stereo(pKF->cameras))        ? pKF->keypoints[i].octave
                                                                                    : std::cmp_less(i, pKF->keypoints.size()) ? pKF->keypoints[i].octave
                                                                                                                              : pKF->right_keypoints[i - pKF->keypoints.size()].octave;
                        const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();
                        int                                          nObs         = 0;
                        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                        {
                            key_frame *pKFi = mit->first;
                            if (pKFi == pKF)
                            {
                                continue;
                            }
                            tuple<int, int> indexes   = mit->second;
                            int             leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int             scaleLeveli = -1;
                            if (!is_fisheye_stereo(pKFi->cameras))
                            {
                                scaleLeveli = pKFi->keypoints[leftIndex].octave;
                            }
                            else
                            {
                                if (leftIndex != -1)
                                {
                                    scaleLeveli = pKFi->keypoints[leftIndex].octave;
                                }
                                if (rightIndex != -1)
                                {
                                    int rightLevel = pKFi->right_keypoints[rightIndex - pKFi->keypoints.size()].octave;
                                    scaleLeveli    = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel : scaleLeveli;
                                }
                            }

                            if (scaleLeveli <= scaleLevel + 1)
                            {
                                nObs++;
                                if (nObs > thObs)
                                {
                                    break;
                                }
                            }
                        }
                        if (nObs > thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if (nRedundantObservations > redundant_th * nMPs)
        {
            if (inertial)
            {
                if (owner_atlas->key_frames_in_map() <= Nd)
                {
                    continue;
                }

                if (pKF->id > (current_key_frame->id - 2))
                {
                    continue;
                }

                if (pKF->previous_key_frame && pKF->next_key_frame)
                {
                    const float t = pKF->next_key_frame->timestamp - pKF->previous_key_frame->timestamp;

                    if ((bInitImu && (pKF->id < last_ID) && t < 3.) || (t < 0.5))
                    {
                        pKF->next_key_frame->imu_preintegrated->merge_previous(pKF->imu_preintegrated);
                        pKF->next_key_frame->previous_key_frame = pKF->previous_key_frame;
                        pKF->previous_key_frame->next_key_frame = pKF->next_key_frame;
                        pKF->next_key_frame                     = NULL;
                        pKF->previous_key_frame                 = NULL;
                        pKF->set_bad_flag();
                    }
                    else if (!current_key_frame->get_map()->get_inertial_ba2() && ((pKF->get_imu_position() - pKF->previous_key_frame->get_imu_position()).norm() < 0.02) && (t < 3))
                    {
                        pKF->next_key_frame->imu_preintegrated->merge_previous(pKF->imu_preintegrated);
                        pKF->next_key_frame->previous_key_frame = pKF->previous_key_frame;
                        pKF->previous_key_frame->next_key_frame = pKF->next_key_frame;
                        pKF->next_key_frame                     = NULL;
                        pKF->previous_key_frame                 = NULL;
                        pKF->set_bad_flag();
                    }
                }
            }
            else
            {
                pKF->set_bad_flag();
            }
        }
        if ((count > 20 && bundle_adjustment_aborted) || count > 100)
        {
            break;
        }
    }
}

void local_mapping::request_reset()
{
    {
        unique_lock<mutex> lock(reset_mutex);
        spdlog::debug("Local mapping map reset requested");
        reset_requested = true;
    }
    spdlog::debug("Local mapping waiting for map reset");

    while (1)
    {
        {
            unique_lock<mutex> lock2(reset_mutex);
            if (!reset_requested)
            {
                break;
            }
        }
        usleep(3000);
    }
    spdlog::debug("Local mapping map reset completed");
}

void local_mapping::request_reset_active_map(map *pMap)
{
    {
        unique_lock<mutex> lock(reset_mutex);
        spdlog::debug("Local mapping active-map reset requested");
        active_map_reset_requested = true;
        map_to_reset               = pMap;
    }
    spdlog::debug("Local mapping waiting for active-map reset");

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
    spdlog::debug("Local mapping active-map reset completed");
}

void local_mapping::reset_if_requested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(reset_mutex);
        if (reset_requested)
        {
            executed_reset = true;

            spdlog::debug("Local mapping resetting Atlas");
            new_key_frames.clear();
            recently_added_map_points.clear();
            reset_requested            = false;
            active_map_reset_requested = false;

            // Inertial parameters
            initialization_elapsed_time = 0.f;
            skip_bundle_adjustment_2    = true;
            skip_bundle_adjustment_1    = true;
            bad_imu                     = false;

            initialization_index = 0;

            spdlog::debug("Local mapping Atlas reset completed");
        }

        if (active_map_reset_requested)
        {
            executed_reset = true;
            spdlog::debug("Local mapping resetting current map");
            new_key_frames.clear();
            recently_added_map_points.clear();

            // Inertial parameters
            initialization_elapsed_time = 0.f;
            skip_bundle_adjustment_2    = true;
            skip_bundle_adjustment_1    = true;
            bad_imu                     = false;

            reset_requested            = false;
            active_map_reset_requested = false;
            spdlog::debug("Local mapping current-map reset completed");
        }
    }
    if (executed_reset)
    {
        spdlog::debug("Local mapping reset mutex released");
    }
}

void local_mapping::request_finish()
{
    unique_lock<mutex> lock(finish_mutex);
    finish_requested = true;
}

bool local_mapping::check_finish()
{
    unique_lock<mutex> lock(finish_mutex);
    return finish_requested;
}

void local_mapping::set_finish()
{
    unique_lock<mutex> lock(finish_mutex);
    finished = true;
    unique_lock<mutex> lock2(stop_mutex);
    stopped = true;
}

bool local_mapping::is_finished()
{
    unique_lock<mutex> lock(finish_mutex);
    return finished;
}

void local_mapping::initialize_imu(float priorG, float priorA, bool bFIBA)
{
    if (reset_requested)
    {
        return;
    }

    float  minTime;
    size_t nMinKF;
    if (monocular)
    {
        minTime = 2.0;
        nMinKF  = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF  = 10;
    }

    if (owner_atlas->key_frames_in_map() < nMinKF)
    {
        return;
    }

    // Retrieve all keyframe in temporal order
    list<key_frame *> lpKF;
    key_frame        *pKF = current_key_frame;
    while (pKF->previous_key_frame)
    {
        lpKF.push_front(pKF);
        pKF = pKF->previous_key_frame;
    }
    lpKF.push_front(pKF);
    vector<key_frame *> vpKF(lpKF.begin(), lpKF.end());

    if (vpKF.size() < nMinKF)
    {
        return;
    }

    first_timestamp = vpKF.front()->timestamp;
    if (current_key_frame->timestamp - first_timestamp < minTime)
    {
        return;
    }

    initializing = true;

    while (check_new_key_frames())
    {
        process_new_key_frame();
        vpKF.push_back(current_key_frame);
        lpKF.push_back(current_key_frame);
    }

    const int N = vpKF.size();
    imu::bias b(0, 0, 0, 0, 0, 0);

    // Compute and KF velocities imu_world_rotation estimation
    if (!current_key_frame->get_map()->is_imu_initialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        for (vector<key_frame *>::iterator itKF = vpKF.begin(); itKF != vpKF.end(); itKF++)
        {
            if (!(*itKF)->imu_preintegrated)
            {
                continue;
            }
            if (!(*itKF)->previous_key_frame)
            {
                continue;
            }

            dirG                 -= (*itKF)->previous_key_frame->get_imu_rotation() * (*itKF)->imu_preintegrated->get_updated_delta_velocity();
            Eigen::Vector3f _vel  = ((*itKF)->get_imu_position() - (*itKF)->previous_key_frame->get_imu_position()) / (*itKF)->imu_preintegrated->delta_time;
            (*itKF)->set_velocity(_vel);
            (*itKF)->previous_key_frame->set_velocity(_vel);
        }

        dirG = dirG / dirG.norm();
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v           = gI.cross(dirG);
        const float     nv          = v.norm();
        const float     cosg        = gI.dot(dirG);
        const float     ang         = acos(cosg);
        Eigen::Vector3f vzg         = v * ang / nv;
        Rwg                         = Sophus::SO3f::exp(vzg).matrix();
        imu_world_rotation          = Rwg.cast<double>();
        initialization_elapsed_time = current_key_frame->timestamp - first_timestamp;
    }
    else
    {
        imu_world_rotation = Eigen::Matrix3d::Identity();
        gyro_bias          = current_key_frame->get_gyro_bias().cast<double>();
        accelerometer_bias = current_key_frame->get_accelerometer_bias().cast<double>();
    }

    scale = 1.0;

    initialization_time = tracker->last_frame.timestamp - vpKF.front()->timestamp;

    optimizer::inertial_optimization(owner_atlas->get_current_map(), imu_world_rotation, scale, gyro_bias, accelerometer_bias, monocular, inertial_information, false, false, priorG, priorA);

    if (scale < 1e-1)
    {
        spdlog::warn("Scale estimate too small");
        initializing = false;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(owner_atlas->get_current_map()->map_update_mutex);
        if ((fabs(scale - 1.f) > 0.00001) || !monocular)
        {
            Sophus::SE3f Twg(imu_world_rotation.cast<float>().transpose(), Eigen::Vector3f::Zero());
            owner_atlas->get_current_map()->apply_scaled_rotation(Twg, scale, true);
            tracker->update_frame_imu(scale, vpKF[0]->get_imu_bias(), current_key_frame);
        }

        // Check if initialization OK
        if (!owner_atlas->is_imu_initialized())
        {
            for (int i = 0; i < N; i++)
            {
                key_frame *pKF2 = vpKF[i];
                pKF2->has_imu   = true;
            }
        }
    }

    tracker->update_frame_imu(1.0, vpKF[0]->get_imu_bias(), current_key_frame);
    if (!owner_atlas->is_imu_initialized())
    {
        owner_atlas->set_imu_initialized();
        tracker->imu_initialization_timestamp = tracker->current_frame.timestamp;
        current_key_frame->has_imu            = true;
    }

    if (bFIBA)
    {
        if (priorA != 0.f)
        {
            optimizer::full_inertial_ba(owner_atlas->get_current_map(), 100, false, current_key_frame->id, NULL, true, priorG, priorA);
        }
        else
        {
            optimizer::full_inertial_ba(owner_atlas->get_current_map(), 100, false, current_key_frame->id, NULL, false);
        }
    }

    spdlog::debug("Global Bundle Adjustment finished\nUpdating map ...");

    // Get Map Mutex
    unique_lock<mutex> lock(owner_atlas->get_current_map()->map_update_mutex);

    unsigned long GBAid = current_key_frame->id;

    // Process keyframes in the queue
    while (check_new_key_frames())
    {
        process_new_key_frame();
        vpKF.push_back(current_key_frame);
        lpKF.push_back(current_key_frame);
    }

    // Correct keyframes starting at map first keyframe
    list<key_frame *> lpKFtoCheck(owner_atlas->get_current_map()->key_frame_origins.begin(), owner_atlas->get_current_map()->key_frame_origins.end());

    while (!lpKFtoCheck.empty())
    {
        key_frame             *pKF     = lpKFtoCheck.front();
        const set<key_frame *> sChilds = pKF->get_children();
        Sophus::SE3f           Twc     = pKF->get_pose_inverse();
        for (set<key_frame *>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++)
        {
            key_frame *pChild = *sit;
            if (!pChild || pChild->is_bad())
            {
                continue;
            }

            if (pChild->global_ba_key_frame_id != GBAid)
            {
                Sophus::SE3f Tchildc          = pChild->get_pose() * Twc;
                pChild->camera_pose_global_ba = Tchildc * pKF->camera_pose_global_ba;

                Sophus::SO3f Rcor = pChild->camera_pose_global_ba.so3().inverse() * pChild->get_pose().so3();
                if (pChild->is_velocity_set())
                {
                    pChild->velocity_world_global_ba = Rcor * pChild->get_velocity();
                }
                else
                {
                    spdlog::debug("Child velocity empty!! ");
                }

                pChild->bias_global_ba         = pChild->get_imu_bias();
                pChild->global_ba_key_frame_id = GBAid;
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
        else
        {
            spdlog::warn("Keyframe {} is not inertial", pKF->id);
        }

        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    const vector<map_point *> vpMPs = owner_atlas->get_current_map()->get_all_map_points();

    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        map_point *pMP = vpMPs[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (pMP->global_ba_key_frame_id == GBAid)
        {
            // If optimized by Global BA, just update
            pMP->set_world_position(pMP->global_ba_position);
        }
        else
        {
            // Update according to the correction of its reference keyframe
            key_frame *pRefKF = pMP->get_reference_key_frame();

            if (pRefKF->global_ba_key_frame_id != GBAid)
            {
                continue;
            }

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->camera_pose_before_global_ba * pMP->get_world_position();

            // Backproject using corrected camera
            pMP->set_world_position(pRefKF->get_pose_inverse() * Xc);
        }
    }

    spdlog::debug("Map updated!");

    key_frame_count = vpKF.size();
    initialization_index++;

    for (list<key_frame *>::iterator lit = new_key_frames.begin(), lend = new_key_frames.end(); lit != lend; lit++)
    {
        (*lit)->set_bad_flag();
        delete *lit;
    }
    new_key_frames.clear();

    tracker->state = tracking::OK;
    initializing   = false;

    current_key_frame->get_map()->increase_change_index();

    return;
}

void local_mapping::refine_scale()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    if (reset_requested)
    {
        return;
    }

    // Retrieve all keyframes in temporal order
    list<key_frame *> lpKF;
    key_frame        *pKF = current_key_frame;
    while (pKF->previous_key_frame)
    {
        lpKF.push_front(pKF);
        pKF = pKF->previous_key_frame;
    }
    lpKF.push_front(pKF);
    vector<key_frame *> vpKF(lpKF.begin(), lpKF.end());

    while (check_new_key_frames())
    {
        process_new_key_frame();
        vpKF.push_back(current_key_frame);
        lpKF.push_back(current_key_frame);
    }

    imu_world_rotation = Eigen::Matrix3d::Identity();
    scale              = 1.0;

    optimizer::inertial_optimization(owner_atlas->get_current_map(), imu_world_rotation, scale);

    if (scale < 1e-1) // 1e-1
    {
        spdlog::warn("Scale estimate too small");
        initializing = false;
        return;
    }

    Sophus::SO3d so3wg(imu_world_rotation);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(owner_atlas->get_current_map()->map_update_mutex);
    if ((fabs(scale - 1.f) > 0.002) || !monocular)
    {
        Sophus::SE3f Tgw(imu_world_rotation.cast<float>().transpose(), Eigen::Vector3f::Zero());
        owner_atlas->get_current_map()->apply_scaled_rotation(Tgw, scale, true);
        tracker->update_frame_imu(scale, current_key_frame->get_imu_bias(), current_key_frame);
    }

    for (list<key_frame *>::iterator lit = new_key_frames.begin(), lend = new_key_frames.end(); lit != lend; lit++)
    {
        (*lit)->set_bad_flag();
        delete *lit;
    }
    new_key_frames.clear();

    // To perform pose-inertial opt w.r.t. last keyframe
    current_key_frame->get_map()->increase_change_index();

    return;
}

bool local_mapping::is_initializing() { return initializing; }

double local_mapping::get_current_key_frame_time()
{

    if (current_key_frame)
    {
        return current_key_frame->timestamp;
    }
    else
    {
        return 0.0;
    }
}

key_frame *local_mapping::get_current_key_frame() { return current_key_frame; }

} // namespace ORB_SLAM3
