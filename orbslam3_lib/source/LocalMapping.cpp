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

#include "LocalMapping.h"

#include "LoopClosing.h"
#include "Optimizer.h"
#include "converter.h"
#include "descriptor_matcher.h"
#include "geometric_tools.h"

#include <mutex>

#include <gsl/gsl>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(processing_statistics &statistics, System *pSys, atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName)
    : mScale(1.0), mInitSect(0), mIdxInit(0), mIdxIteration(0), mbNotBA1(true), mbNotBA2(true), _statistics(statistics), mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial),
      mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), mbAbortBA(false), mbStopped(false), mbStopRequested(false),
      mbNotStop(false), mbAcceptKeyFrames(true), bInitializing(false), infoInertial(Eigen::MatrixXd::Zero(9, 9))
{
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM        = 0;
    mNumKFCulling = 0;
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser) { mpLoopCloser = pLoopCloser; }

void LocalMapping::SetTracker(Tracking *pTracker) { mpTracker = pTracker; }

void LocalMapping::Run()
{
    mbFinished = false;

    while (1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if (CheckNewKeyFrames() && !mbBadImu)
        {
            const processing_statistics::backend_timer statistics_timer(_statistics, processing_statistics::backend_stage::LOCAL_MAPPING);
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            CreateNewMapPoints();

            mbAbortBA = false;

            if (!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            [[maybe_unused]] bool b_doneLBA      = false;
            int                   num_FixedKF_BA = 0;
            int                   num_OptKF_BA   = 0;
            int                   num_MPs_BA     = 0;
            int                   num_edges_BA   = 0;

            if (!CheckNewKeyFrames() && !stopRequested())
            {
                if (mpAtlas->key_frames_in_map() > 2)
                {

                    if (mbInertial && mpCurrentKeyFrame->get_map()->is_imu_initialized())
                    {
                        float dist = (mpCurrentKeyFrame->previous_key_frame->get_camera_center() - mpCurrentKeyFrame->get_camera_center()).norm() +
                                     (mpCurrentKeyFrame->previous_key_frame->previous_key_frame->get_camera_center() - mpCurrentKeyFrame->previous_key_frame->get_camera_center()).norm();

                        if (dist > 0.05)
                        {
                            mTinit += mpCurrentKeyFrame->timestamp - mpCurrentKeyFrame->previous_key_frame->timestamp;
                        }
                        if (!mpCurrentKeyFrame->get_map()->get_inertial_ba2())
                        {
                            if ((mTinit < 10.f) && (dist < 0.02))
                            {
                                spdlog::warn("Not enough motion for initialization; resetting active map");
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true;
                                mpMapToReset              = mpCurrentKeyFrame->get_map();
                                mbBadImu                  = true;
                            }
                        }

                        bool bLarge = ((mpTracker->GetMatchesInliers() > 75) && mbMonocular) || ((mpTracker->GetMatchesInliers() > 100) && !mbMonocular);
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame,
                                                   &mbAbortBA,
                                                   mpCurrentKeyFrame->get_map(),
                                                   num_FixedKF_BA,
                                                   num_OptKF_BA,
                                                   num_MPs_BA,
                                                   num_edges_BA,
                                                   bLarge,
                                                   !mpCurrentKeyFrame->get_map()->get_inertial_ba2());
                        b_doneLBA = true;
                    }
                    else
                    {
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->get_map(), num_FixedKF_BA, num_OptKF_BA, num_MPs_BA, num_edges_BA);
                        b_doneLBA = true;
                    }
                }

                // Initialize IMU here
                if (!mpCurrentKeyFrame->get_map()->is_imu_initialized() && mbInertial)
                {
                    if (mbMonocular)
                    {
                        InitializeIMU(1e2, 1e10, true);
                    }
                    else
                    {
                        InitializeIMU(1e2, 1e5, true);
                    }
                }

                // Check redundant local Keyframes
                KeyFrameCulling();

                if ((mTinit < 50.0f) && mbInertial)
                {
                    if (mpCurrentKeyFrame->get_map()->is_imu_initialized() && mpTracker->mState == Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        if (!mpCurrentKeyFrame->get_map()->get_inertial_ba1())
                        {
                            if (mTinit > 5.0f)
                            {
                                spdlog::info("Starting VIBA stage 1");
                                mpCurrentKeyFrame->get_map()->set_inertial_ba1();
                                if (mbMonocular)
                                {
                                    InitializeIMU(1.f, 1e5, true);
                                }
                                else
                                {
                                    InitializeIMU(1.f, 1e5, true);
                                }

                                spdlog::info("Finished VIBA stage 1");
                            }
                        }
                        else if (!mpCurrentKeyFrame->get_map()->get_inertial_ba2())
                        {
                            if (mTinit > 15.0f)
                            {
                                spdlog::info("Starting VIBA stage 2");
                                mpCurrentKeyFrame->get_map()->set_inertial_ba2();
                                if (mbMonocular)
                                {
                                    InitializeIMU(0.f, 0.f, true);
                                }
                                else
                                {
                                    InitializeIMU(0.f, 0.f, true);
                                }

                                spdlog::info("Finished VIBA stage 2");
                            }
                        }

                        // scale refinement
                        if (((mpAtlas->key_frames_in_map()) <= 200) && ((mTinit > 25.0f && mTinit < 25.5f) || (mTinit > 35.0f && mTinit < 35.5f) || (mTinit > 45.0f && mTinit < 45.5f) ||
                                                                        (mTinit > 55.0f && mTinit < 55.5f) || (mTinit > 65.0f && mTinit < 65.5f) || (mTinit > 75.0f && mTinit < 75.5f)))
                        {
                            if (mbMonocular)
                            {
                                ScaleRefinement();
                            }
                        }
                    }
                }
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if (Stop() && !mbBadImu)
        {
            // Safe area to stop
            while (isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if (CheckFinish())
            {
                break;
            }
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if (CheckFinish())
        {
            break;
        }

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(key_frame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->compute_bow();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<map_point *> vpMapPointMatches = mpCurrentKeyFrame->get_map_point_matches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        map_point *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->is_bad())
            {
                if (!pMP->is_in_key_frame(mpCurrentKeyFrame))
                {
                    pMP->add_observation(mpCurrentKeyFrame, i);
                    pMP->update_normal_and_depth();
                    pMP->compute_distinctive_descriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->update_connections();

    // Insert Keyframe in Map
    mpAtlas->add_key_frame(mpCurrentKeyFrame);
}

void LocalMapping::EmptyQueue()
{
    while (CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
    }
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<map_point *>::iterator lit          = mlpRecentAddedMapPoints.begin();
    const unsigned long int     nCurrentKFid = mpCurrentKeyFrame->id;

    int nThObs;
    if (mbMonocular)
    {
        nThObs = 2;
    }
    else
    {
        nThObs = 3;
    }
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size();

    while (lit != mlpRecentAddedMapPoints.end())
    {
        map_point *pMP = *lit;

        if (pMP->is_bad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (pMP->get_found_ratio() < 0.25f)
        {
            pMP->set_bad_flag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->first_key_frame_id) >= 2 && pMP->get_observation_count() <= cnThObs)
        {
            pMP->set_bad_flag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->first_key_frame_id) >= 3)
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
            borrar--;
        }
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // For stereo inertial case
    if (mbMonocular)
    {
        nn = 30;
    }
    vector<key_frame *> vpNeighKFs = mpCurrentKeyFrame->get_best_covisibility_key_frames(nn);

    if (mbInertial)
    {
        key_frame *pKF   = mpCurrentKeyFrame;
        int       count = 0;
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

    Sophus::SE3<float>         sophTcw1 = mpCurrentKeyFrame->get_pose();
    Eigen::Matrix<float, 3, 4> eigTcw1  = sophTcw1.matrix3x4();
    Eigen::Matrix<float, 3, 3> Rcw1     = eigTcw1.block<3, 3>(0, 0);
    Eigen::Matrix<float, 3, 3> Rwc1     = Rcw1.transpose();
    Eigen::Vector3f            tcw1     = sophTcw1.translation();
    Eigen::Vector3f            Ow1      = mpCurrentKeyFrame->get_camera_center();

    const float                  &fx1    = mpCurrentKeyFrame->fx;
    const float                  &fy1    = mpCurrentKeyFrame->fy;
    const float                  &cx1    = mpCurrentKeyFrame->cx;
    const float                  &cy1    = mpCurrentKeyFrame->cy;
    [[maybe_unused]] const float &invfx1 = mpCurrentKeyFrame->invfx;
    [[maybe_unused]] const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor         = 1.5f * mpCurrentKeyFrame->scale_factor;
    int         countStereo         = 0;
    int         countStereoGoodProj = 0;
    int         countStereoAttempt  = 0;
    int         totalStereoPts      = 0;
    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        if (i > 0 && CheckNewKeyFrames())
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

        if (!mbMonocular)
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
        bool                         bCoarse = mbInertial && mpTracker->mState == Tracking::RECENTLY_LOST && mpCurrentKeyFrame->get_map()->get_inertial_ba2();

        matcher.search_for_triangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, false, bCoarse);

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

            const cv::KeyPoint &kp1 = ((!is_fisheye_stereo(mpCurrentKeyFrame->cameras))          ? mpCurrentKeyFrame->keypoints[idx1]
                                       : (std::cmp_less(idx1, mpCurrentKeyFrame->keypoints.size())) ? mpCurrentKeyFrame->keypoints[idx1]
                                                                                                 : mpCurrentKeyFrame->right_keypoints[idx1 - mpCurrentKeyFrame->keypoints.size()])
                                          .undistorted();
            const float kp1_ur   = mpCurrentKeyFrame->right_coordinates[idx1];
            bool        bStereo1 = (!is_fisheye_stereo(mpCurrentKeyFrame->cameras) && kp1_ur >= 0);
            const bool  bRight1  = is_fisheye_stereo(mpCurrentKeyFrame->cameras) && std::cmp_greater_equal(idx1, mpCurrentKeyFrame->keypoints.size());

            const cv::KeyPoint &kp2 = ((!is_fisheye_stereo(pKF2->cameras))        ? pKF2->keypoints[idx2]
                                       : std::cmp_less(idx2, pKF2->keypoints.size()) ? pKF2->keypoints[idx2]
                                                                                  : pKF2->right_keypoints[idx2 - pKF2->keypoints.size()])
                                          .undistorted();

            const float kp2_ur   = pKF2->right_coordinates[idx2];
            bool        bStereo2 = (!is_fisheye_stereo(pKF2->cameras) && kp2_ur >= 0);
            const bool  bRight2  = is_fisheye_stereo(pKF2->cameras) && std::cmp_greater_equal(idx2, pKF2->keypoints.size());

            if (is_fisheye_stereo(mpCurrentKeyFrame->cameras) && is_fisheye_stereo(pKF2->cameras))
            {
                if (bRight1 && bRight2)
                {
                    sophTcw1 = mpCurrentKeyFrame->get_right_pose();
                    Ow1      = mpCurrentKeyFrame->get_right_camera_center();

                    sophTcw2 = pKF2->get_right_pose();
                    Ow2      = pKF2->get_right_camera_center();

                    camera_index1 = 1;
                    camera_index2 = 1;
                }
                else if (bRight1 && !bRight2)
                {
                    sophTcw1 = mpCurrentKeyFrame->get_right_pose();
                    Ow1      = mpCurrentKeyFrame->get_right_camera_center();

                    sophTcw2 = pKF2->get_pose();
                    Ow2      = pKF2->get_camera_center();

                    camera_index1 = 1;
                    camera_index2 = 0;
                }
                else if (!bRight1 && bRight2)
                {
                    sophTcw1 = mpCurrentKeyFrame->get_pose();
                    Ow1      = mpCurrentKeyFrame->get_camera_center();

                    sophTcw2 = pKF2->get_right_pose();
                    Ow2      = pKF2->get_right_camera_center();

                    camera_index1 = 0;
                    camera_index2 = 1;
                }
                else
                {
                    sophTcw1 = mpCurrentKeyFrame->get_pose();
                    Ow1      = mpCurrentKeyFrame->get_camera_center();

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
            Eigen::Vector3f xn1 = mpCurrentKeyFrame->cameras[camera_index1]->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pKF2->cameras[camera_index2]->unprojectEig(kp2.pt);

            Eigen::Vector3f ray1            = Rwc1 * xn1;
            Eigen::Vector3f ray2            = Rwc2 * xn2;
            const float     cosParallaxRays = ray1.dot(ray2) / (ray1.norm() * ray2.norm());

            float cosParallaxStereo  = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1)
            {
                cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->depths[idx1]));
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
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 || (cosParallaxRays < 0.9996 && mbInertial) || (cosParallaxRays < 0.9998 && !mbInertial)))
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
                goodProj     = mpCurrentKeyFrame->unproject_stereo(idx1, x3D);
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
            const float &sigmaSquare1 = mpCurrentKeyFrame->level_sigma_squared[kp1.octave];
            const float  x1           = Rcw1.row(0).dot(x3D) + tcw1(0);
            const float  y1           = Rcw1.row(1).dot(x3D) + tcw1(1);
            const float  invz1        = 1.0 / z1;

            if (!bStereo1)
            {
                cv::Point2f uv1   = mpCurrentKeyFrame->cameras[camera_index1]->project(cv::Point3f(x1, y1, z1));
                float       errX1 = uv1.x - kp1.pt.x;
                float       errY1 = uv1.y - kp1.pt.y;

                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                {
                    continue;
                }
            }
            else
            {
                float u1      = fx1 * x1 * invz1 + cx1;
                float u1_r    = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1      = fy1 * y1 * invz1 + cy1;
                float errX1   = u1 - kp1.pt.x;
                float errY1   = v1 - kp1.pt.y;
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
                float       errX2 = uv2.x - kp2.pt.x;
                float       errY2 = uv2.y - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                {
                    continue;
                }
            }
            else
            {
                float u2      = fx2 * x2 * invz2 + cx2;
                float u2_r    = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2      = fy2 * y2 * invz2 + cy2;
                float errX2   = u2 - kp2.pt.x;
                float errY2   = v2 - kp2.pt.y;
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

            if (mbFarPoints && (dist1 >= mThFarPoints || dist2 >= mThFarPoints)) // MODIFICATION
            {
                continue;
            }

            const float ratioDist   = dist2 / dist1;
            const float ratioOctave = mpCurrentKeyFrame->scale_factors[kp1.octave] / pKF2->scale_factors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
            {
                continue;
            }

            // Triangulation is succesfull
            map_point *pMP = new map_point(x3D, mpCurrentKeyFrame, mpAtlas->get_current_map());
            if (bPointStereo)
            {
                countStereo++;
            }

            pMP->add_observation(mpCurrentKeyFrame, idx1);
            pMP->add_observation(pKF2, idx2);

            mpCurrentKeyFrame->add_map_point(pMP, idx1);
            pKF2->add_map_point(pMP, idx2);

            pMP->compute_distinctive_descriptors();

            pMP->update_normal_and_depth();

            mpAtlas->add_map_point(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if (mbMonocular)
    {
        nn = 30;
    }
    const vector<key_frame *> vpNeighKFs = mpCurrentKeyFrame->get_best_covisibility_key_frames(nn);
    vector<key_frame *>       vpTargetKFs;
    for (vector<key_frame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        key_frame *pKFi = *vit;
        if (pKFi->is_bad() || pKFi->fuse_target_key_frame_id == mpCurrentKeyFrame->id)
        {
            continue;
        }
        vpTargetKFs.push_back(pKFi);
        pKFi->fuse_target_key_frame_id = mpCurrentKeyFrame->id;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for (int i = 0, imax = vpTargetKFs.size(); i < imax; i++)
    {
        const vector<key_frame *> vpSecondNeighKFs = vpTargetKFs[i]->get_best_covisibility_key_frames(20);
        for (vector<key_frame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            key_frame *pKFi2 = *vit2;
            if (pKFi2->is_bad() || pKFi2->fuse_target_key_frame_id == mpCurrentKeyFrame->id || pKFi2->id == mpCurrentKeyFrame->id)
            {
                continue;
            }
            vpTargetKFs.push_back(pKFi2);
            pKFi2->fuse_target_key_frame_id = mpCurrentKeyFrame->id;
        }
        if (mbAbortBA)
        {
            break;
        }
    }

    // Extend to temporal neighbors
    if (mbInertial)
    {
        key_frame *pKFi = mpCurrentKeyFrame->previous_key_frame;
        while (vpTargetKFs.size() < 20 && pKFi)
        {
            if (pKFi->is_bad() || pKFi->fuse_target_key_frame_id == mpCurrentKeyFrame->id)
            {
                pKFi = pKFi->previous_key_frame;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->fuse_target_key_frame_id = mpCurrentKeyFrame->id;
            pKFi                    = pKFi->previous_key_frame;
        }
    }

    // Search matches by projection from current KF in target KFs
    descriptor_matcher  matcher;
    vector<map_point *> vpMapPointMatches = mpCurrentKeyFrame->get_map_point_matches();
    for (vector<key_frame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
    {
        key_frame *pKFi = *vit;

        matcher.Fuse(pKFi, vpMapPointMatches);
        if (is_fisheye_stereo(pKFi->cameras))
        {
            matcher.Fuse(pKFi, vpMapPointMatches, true);
        }
    }

    if (mbAbortBA)
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
            if (pMP->is_bad() || pMP->fuse_candidate_key_frame_id == mpCurrentKeyFrame->id)
            {
                continue;
            }
            pMP->fuse_candidate_key_frame_id = mpCurrentKeyFrame->id;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);
    if (is_fisheye_stereo(mpCurrentKeyFrame->cameras))
    {
        matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates, true);
    }

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->get_map_point_matches();
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
    mpCurrentKeyFrame->update_connections();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        spdlog::debug("Local mapping stopped");
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if (mbFinished)
    {
        return;
    }
    mbStopped       = false;
    mbStopRequested = false;
    for (list<key_frame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
    {
        delete *lit;
    }
    mlNewKeyFrames.clear();

    spdlog::debug("Local mapping released");
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
    {
        return false;
    }

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA() { mbAbortBA = true; }

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->update_best_covisibles();
    vector<key_frame *> vpLocalKeyFrames = mpCurrentKeyFrame->get_vector_covisible_key_frames();

    float redundant_th;
    if (!mbInertial)
    {
        redundant_th = 0.9;
    }
    else if (mbMonocular)
    {
        redundant_th = 0.9;
    }
    else
    {
        redundant_th = 0.5;
    }

    const bool bInitImu = mpAtlas->is_imu_initialized();
    int        count    = 0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID = 0;
    if (mbInertial)
    {
        int       count  = 0;
        key_frame *aux_KF = mpCurrentKeyFrame;
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
                    if (!mbMonocular)
                    {
                        if (pKF->depths[i] > pKF->depth_threshold || pKF->depths[i] < 0)
                        {
                            continue;
                        }
                    }

                    nMPs++;
                    if (pMP->get_observation_count() > thObs)
                    {
                        const int                                  &scaleLevel   = (!is_fisheye_stereo(pKF->cameras))     ? pKF->keypoints[i].octave
                                                                                   : std::cmp_less(i, pKF->keypoints.size()) ? pKF->keypoints[i].octave
                                                                                                                          : pKF->right_keypoints[i - pKF->keypoints.size()].octave;
                        const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();
                        int                                         nObs         = 0;
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
            if (mbInertial)
            {
                if (mpAtlas->key_frames_in_map() <= Nd)
                {
                    continue;
                }

                if (pKF->id > (mpCurrentKeyFrame->id - 2))
                {
                    continue;
                }

                if (pKF->previous_key_frame && pKF->next_key_frame)
                {
                    const float t = pKF->next_key_frame->timestamp - pKF->previous_key_frame->timestamp;

                    if ((bInitImu && (pKF->id < last_ID) && t < 3.) || (t < 0.5))
                    {
                        pKF->next_key_frame->imu_preintegrated->MergePrevious(pKF->imu_preintegrated);
                        pKF->next_key_frame->previous_key_frame = pKF->previous_key_frame;
                        pKF->previous_key_frame->next_key_frame = pKF->next_key_frame;
                        pKF->next_key_frame          = NULL;
                        pKF->previous_key_frame          = NULL;
                        pKF->set_bad_flag();
                    }
                    else if (!mpCurrentKeyFrame->get_map()->get_inertial_ba2() && ((pKF->get_imu_position() - pKF->previous_key_frame->get_imu_position()).norm() < 0.02) && (t < 3))
                    {
                        pKF->next_key_frame->imu_preintegrated->MergePrevious(pKF->imu_preintegrated);
                        pKF->next_key_frame->previous_key_frame = pKF->previous_key_frame;
                        pKF->previous_key_frame->next_key_frame = pKF->next_key_frame;
                        pKF->next_key_frame          = NULL;
                        pKF->previous_key_frame          = NULL;
                        pKF->set_bad_flag();
                    }
                }
            }
            else
            {
                pKF->set_bad_flag();
            }
        }
        if ((count > 20 && mbAbortBA) || count > 100)
        {
            break;
        }
    }
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        spdlog::debug("Local mapping map reset requested");
        mbResetRequested = true;
    }
    spdlog::debug("Local mapping waiting for map reset");

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
            {
                break;
            }
        }
        usleep(3000);
    }
    spdlog::debug("Local mapping map reset completed");
}

void LocalMapping::RequestResetActiveMap(map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        spdlog::debug("Local mapping active-map reset requested");
        mbResetRequestedActiveMap = true;
        mpMapToReset              = pMap;
    }
    spdlog::debug("Local mapping waiting for active-map reset");

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequestedActiveMap)
            {
                break;
            }
        }
        usleep(3000);
    }
    spdlog::debug("Local mapping active-map reset completed");
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbResetRequested)
        {
            executed_reset = true;

            spdlog::debug("Local mapping resetting Atlas");
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested          = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit   = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu = false;

            mIdxInit = 0;

            spdlog::debug("Local mapping Atlas reset completed");
        }

        if (mbResetRequestedActiveMap)
        {
            executed_reset = true;
            spdlog::debug("Local mapping resetting current map");
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit   = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu = false;

            mbResetRequested          = false;
            mbResetRequestedActiveMap = false;
            spdlog::debug("Local mapping current-map reset completed");
        }
    }
    if (executed_reset)
    {
        spdlog::debug("Local mapping reset mutex released");
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested)
    {
        return;
    }

    float  minTime;
    size_t nMinKF;
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF  = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF  = 10;
    }

    if (mpAtlas->key_frames_in_map() < nMinKF)
    {
        return;
    }

    // Retrieve all keyframe in temporal order
    list<key_frame *> lpKF;
    key_frame        *pKF = mpCurrentKeyFrame;
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

    mFirstTs = vpKF.front()->timestamp;
    if (mpCurrentKeyFrame->timestamp - mFirstTs < minTime)
    {
        return;
    }

    bInitializing = true;

    while (CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0, 0, 0, 0, 0, 0);

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->get_map()->is_imu_initialized())
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

            dirG                 -= (*itKF)->previous_key_frame->get_imu_rotation() * (*itKF)->imu_preintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3f _vel  = ((*itKF)->get_imu_position() - (*itKF)->previous_key_frame->get_imu_position()) / (*itKF)->imu_preintegrated->dT;
            (*itKF)->set_velocity(_vel);
            (*itKF)->previous_key_frame->set_velocity(_vel);
        }

        dirG = dirG / dirG.norm();
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v    = gI.cross(dirG);
        const float     nv   = v.norm();
        const float     cosg = gI.dot(dirG);
        const float     ang  = acos(cosg);
        Eigen::Vector3f vzg  = v * ang / nv;
        Rwg                  = Sophus::SO3f::exp(vzg).matrix();
        mRwg                 = Rwg.cast<double>();
        mTinit               = mpCurrentKeyFrame->timestamp - mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg  = mpCurrentKeyFrame->get_gyro_bias().cast<double>();
        mba  = mpCurrentKeyFrame->get_accelerometer_bias().cast<double>();
    }

    mScale = 1.0;

    mInitTime = mpTracker->mLastFrame.timestamp - vpKF.front()->timestamp;

    Optimizer::InertialOptimization(mpAtlas->get_current_map(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);

    if (mScale < 1e-1)
    {
        spdlog::warn("Scale estimate too small");
        bInitializing = false;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->get_current_map()->map_update_mutex);
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular)
        {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpAtlas->get_current_map()->apply_scaled_rotation(Twg, mScale, true);
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->get_imu_bias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        if (!mpAtlas->is_imu_initialized())
        {
            for (int i = 0; i < N; i++)
            {
                key_frame *pKF2 = vpKF[i];
                pKF2->has_imu     = true;
            }
        }
    }

    mpTracker->UpdateFrameIMU(1.0, vpKF[0]->get_imu_bias(), mpCurrentKeyFrame);
    if (!mpAtlas->is_imu_initialized())
    {
        mpAtlas->set_imu_initialized();
        mpTracker->t0IMU        = mpTracker->mCurrentFrame.timestamp;
        mpCurrentKeyFrame->has_imu = true;
    }

    if (bFIBA)
    {
        if (priorA != 0.f)
        {
            Optimizer::FullInertialBA(mpAtlas->get_current_map(), 100, false, mpCurrentKeyFrame->id, NULL, true, priorG, priorA);
        }
        else
        {
            Optimizer::FullInertialBA(mpAtlas->get_current_map(), 100, false, mpCurrentKeyFrame->id, NULL, false);
        }
    }

    spdlog::debug("Global Bundle Adjustment finished\nUpdating map ...");

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->get_current_map()->map_update_mutex);

    unsigned long GBAid = mpCurrentKeyFrame->id;

    // Process keyframes in the queue
    while (CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    list<key_frame *> lpKFtoCheck(mpAtlas->get_current_map()->key_frame_origins.begin(), mpAtlas->get_current_map()->key_frame_origins.end());

    while (!lpKFtoCheck.empty())
    {
        key_frame             *pKF     = lpKFtoCheck.front();
        const set<key_frame *> sChilds = pKF->get_children();
        Sophus::SE3f          Twc     = pKF->get_pose_inverse();
        for (set<key_frame *>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++)
        {
            key_frame *pChild = *sit;
            if (!pChild || pChild->is_bad())
            {
                continue;
            }

            if (pChild->global_ba_key_frame_id != GBAid)
            {
                Sophus::SE3f Tchildc = pChild->get_pose() * Twc;
                pChild->camera_pose_global_ba      = Tchildc * pKF->camera_pose_global_ba;

                Sophus::SO3f Rcor = pChild->camera_pose_global_ba.so3().inverse() * pChild->get_pose().so3();
                if (pChild->is_velocity_set())
                {
                    pChild->velocity_world_global_ba = Rcor * pChild->get_velocity();
                }
                else
                {
                    spdlog::debug("Child velocity empty!! ");
                }

                pChild->bias_global_ba        = pChild->get_imu_bias();
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
    const vector<map_point *> vpMPs = mpAtlas->get_current_map()->get_all_map_points();

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

    mnKFs = vpKF.size();
    mIdxInit++;

    for (list<key_frame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
    {
        (*lit)->set_bad_flag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState = Tracking::OK;
    bInitializing     = false;

    mpCurrentKeyFrame->get_map()->increase_change_index();

    return;
}

void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    if (mbResetRequested)
    {
        return;
    }

    // Retrieve all keyframes in temporal order
    list<key_frame *> lpKF;
    key_frame        *pKF = mpCurrentKeyFrame;
    while (pKF->previous_key_frame)
    {
        lpKF.push_front(pKF);
        pKF = pKF->previous_key_frame;
    }
    lpKF.push_front(pKF);
    vector<key_frame *> vpKF(lpKF.begin(), lpKF.end());

    while (CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    mRwg   = Eigen::Matrix3d::Identity();
    mScale = 1.0;

    Optimizer::InertialOptimization(mpAtlas->get_current_map(), mRwg, mScale);

    if (mScale < 1e-1) // 1e-1
    {
        spdlog::warn("Scale estimate too small");
        bInitializing = false;
        return;
    }

    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->get_current_map()->map_update_mutex);
    if ((fabs(mScale - 1.f) > 0.002) || !mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
        mpAtlas->get_current_map()->apply_scaled_rotation(Tgw, mScale, true);
        mpTracker->UpdateFrameIMU(mScale, mpCurrentKeyFrame->get_imu_bias(), mpCurrentKeyFrame);
    }

    for (list<key_frame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
    {
        (*lit)->set_bad_flag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->get_map()->increase_change_index();

    return;
}

bool LocalMapping::IsInitializing() { return bInitializing; }

double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->timestamp;
    }
    else
    {
        return 0.0;
    }
}

key_frame *LocalMapping::GetCurrKF() { return mpCurrentKeyFrame; }

} // namespace ORB_SLAM3
