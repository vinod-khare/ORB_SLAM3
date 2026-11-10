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

#include "LoopClosing.h"

#include "G2oTypes.h"
#include "Optimizer.h"
#include "converter.h"
#include "descriptor_matcher.h"
#include "sim3_solver.h"

#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

LoopClosing::LoopClosing(processing_statistics &statistics, atlas *pAtlas, key_frame_database *pDB, place_vocabulary *pVoc, const bool bFixScale, const bool bActiveLC)
    : _statistics(statistics), mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), mpKeyFrameDB(pDB),
      place_vocabulary_ptr(pVoc), mpMatchedKF(NULL), mbLoopDetected(false), mnLoopNumCoincidences(0), mnLoopNumNotFound(0), mbMergeDetected(false), mnMergeNumCoincidences(0),
      mnMergeNumNotFound(0), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true), mbStopGBA(false), mbFixScale(bFixScale), mnFullBAIdx(0), mbActiveLC(bActiveLC)
{
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF             = static_cast<key_frame *>(NULL);
    mstrFolderSubTraj           = "SubTrajectories/";
    mnNumCorrection             = 0;
    mnCorrectionGBA             = 0;
}

LoopClosing::~LoopClosing() { stop_global_bundle_adjustment(); }

void LoopClosing::SetTracker(Tracking *pTracker) { mpTracker = pTracker; }

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper) { mpLocalMapper = pLocalMapper; }

void LoopClosing::Run()
{
    mbFinished = false;

    while (1)
    {

        // NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------

        if (CheckNewKeyFrames())
        {
            const processing_statistics::backend_timer statistics_timer(_statistics, processing_statistics::backend_stage::LOOP_CLOSING);
            if (mpLastCurrentKF)
            {
                mpLastCurrentKF->loop_candidate_key_frames.clear();
                mpLastCurrentKF->merge_candidate_key_frames.clear();
            }

            bool bFindedRegion = NewDetectCommonRegions();

            if (bFindedRegion)
            {
                if (mbMergeDetected)
                {
                    if ((mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD) &&
                        (!mpCurrentKF->get_map()->is_imu_initialized()))
                    {
                        spdlog::warn("IMU is not initialized; aborting merge");
                    }
                    else
                    {
                        Sophus::SE3d               mTmw = mpMergeMatchedKF->get_pose().cast<double>();
                        g2o::Sim3                  gSmw2(mTmw.unit_quaternion(), mTmw.translation(), 1.0);
                        Sophus::SE3d               mTcw = mpCurrentKF->get_pose().cast<double>();
                        g2o::Sim3                  gScw1(mTcw.unit_quaternion(), mTcw.translation(), 1.0);
                        g2o::Sim3                  gSw2c = mg2oMergeSlw.inverse();
                        [[maybe_unused]] g2o::Sim3 gSw1m = mg2oMergeSlw;

                        mSold_new = (gSw2c * gScw1);

                        if (mpCurrentKF->get_map()->is_inertial() && mpMergeMatchedKF->get_map()->is_inertial())
                        {
                            spdlog::debug("Checking merge transformation with IMU");
                            if (mSold_new.scale() < 0.90 || mSold_new.scale() > 1.1)
                            {
                                mpMergeLastCurrentKF->set_erase();
                                mpMergeMatchedKF->set_erase();
                                mnMergeNumCoincidences = 0;
                                mvpMergeMatchedMPs.clear();
                                mvpMergeMPs.clear();
                                mnMergeNumNotFound = 0;
                                mbMergeDetected    = false;
                                spdlog::warn("Scale estimate is outside merge bounds; aborting merge");
                                continue;
                            }
                            // If inertial, force only yaw
                            if ((mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD) &&
                                mpCurrentKF->get_map()->get_inertial_ba1())
                            {
                                Eigen::Vector3d phi = LogSO3(mSold_new.rotation().toRotationMatrix());
                                phi(0)              = 0;
                                phi(1)              = 0;
                                mSold_new           = g2o::Sim3(ExpSO3(phi), mSold_new.translation(), 1.0);
                            }
                        }

                        mg2oMergeSmw = gSmw2 * gSw2c * gScw1;

                        mg2oMergeScw = mg2oMergeSlw;

                        // mpTracker->SetStepByStep(true);

                        spdlog::info("Merge detected");

                        // TODO UNCOMMENT
                        if (mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD)
                        {
                            MergeLocal2();
                        }
                        else
                        {
                            MergeLocal();
                        }

                        spdlog::info("Merge finished!");
                    }

                    vdPR_CurrentTime.push_back(mpCurrentKF->timestamp);
                    vdPR_MatchedTime.push_back(mpMergeMatchedKF->timestamp);
                    vnPR_TypeRecogn.push_back(1);

                    // Reset all variables
                    mpMergeLastCurrentKF->set_erase();
                    mpMergeMatchedKF->set_erase();
                    mnMergeNumCoincidences = 0;
                    mvpMergeMatchedMPs.clear();
                    mvpMergeMPs.clear();
                    mnMergeNumNotFound = 0;
                    mbMergeDetected    = false;

                    if (mbLoopDetected)
                    {
                        // Reset Loop variables
                        mpLoopLastCurrentKF->set_erase();
                        mpLoopMatchedKF->set_erase();
                        mnLoopNumCoincidences = 0;
                        mvpLoopMatchedMPs.clear();
                        mvpLoopMPs.clear();
                        mnLoopNumNotFound = 0;
                        mbLoopDetected    = false;
                    }
                }

                if (mbLoopDetected)
                {
                    bool bGoodLoop = true;
                    vdPR_CurrentTime.push_back(mpCurrentKF->timestamp);
                    vdPR_MatchedTime.push_back(mpLoopMatchedKF->timestamp);
                    vnPR_TypeRecogn.push_back(0);

                    spdlog::info("*Loop detected");

                    mg2oLoopScw = mg2oLoopSlw; //*mvg2oSim3LoopTcw[nCurrentIndex];
                    if (mpCurrentKF->get_map()->is_inertial())
                    {
                        Sophus::SE3d Twc = mpCurrentKF->get_pose_inverse().cast<double>();
                        g2o::Sim3    g2oTwc(Twc.unit_quaternion(), Twc.translation(), 1.0);
                        g2o::Sim3    g2oSww_new = g2oTwc * mg2oLoopScw;

                        Eigen::Vector3d    phi = LogSO3(g2oSww_new.rotation().toRotationMatrix());
                        std::ostringstream phi_output;
                        phi_output << phi.transpose();
                        spdlog::debug("Loop rotation phi = {}", phi_output.str());
                        if (fabs(phi(0)) < 0.008f && fabs(phi(1)) < 0.008f && fabs(phi(2)) < 0.349f)
                        {
                            if (mpCurrentKF->get_map()->is_inertial())
                            {
                                // If inertial, force only yaw
                                if ((mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD) &&
                                    mpCurrentKF->get_map()->get_inertial_ba2())
                                {
                                    phi(0)      = 0;
                                    phi(1)      = 0;
                                    g2oSww_new  = g2o::Sim3(ExpSO3(phi), g2oSww_new.translation(), 1.0);
                                    mg2oLoopScw = g2oTwc.inverse() * g2oSww_new;
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

                        mvpLoopMapPoints = mvpLoopMPs;

                        CorrectLoop();

                        mnNumCorrection += 1;
                    }

                    // Reset all variables
                    mpLoopLastCurrentKF->set_erase();
                    mpLoopMatchedKF->set_erase();
                    mnLoopNumCoincidences = 0;
                    mvpLoopMatchedMPs.clear();
                    mvpLoopMPs.clear();
                    mnLoopNumNotFound = 0;
                    mbLoopDetected    = false;
                }
            }
            mpLastCurrentKF = mpCurrentKF;
        }

        ResetIfRequested();

        if (CheckFinish())
        {
            break;
        }

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(key_frame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if (pKF->id != 0)
    {
        mlpLoopKeyFrameQueue.push_back(pKF);
    }
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return (!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::NewDetectCommonRegions()
{
    // To deactivate placerecognition. No loopclosing nor merging will be performed
    if (!mbActiveLC)
    {
        return false;
    }

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->set_not_erase();
        mpCurrentKF->current_place_recognition = true;

        mpLastMap = mpCurrentKF->get_map();
    }

    if (mpLastMap->is_inertial() && !mpLastMap->get_inertial_ba2())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->set_erase();
        return false;
    }

    if (mpTracker->mSensor == System::STEREO && mpLastMap->get_all_key_frames().size() < 5) // 12
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->set_erase();
        return false;
    }

    if (mpLastMap->get_all_key_frames().size() < 12)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->set_erase();
        return false;
    }

    // Check the last candidates with geometric validation
    //  Loop candidates
    bool                  bLoopDetectedInKF = false;
    [[maybe_unused]] bool bCheckSpatial     = false;

    if (mnLoopNumCoincidences > 0)
    {
        bCheckSpatial = true;
        // Find from the last KF candidates
        Sophus::SE3d        mTcl = (mpCurrentKF->get_pose() * mpLoopLastCurrentKF->get_pose_inverse()).cast<double>();
        g2o::Sim3           gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3           gScw           = gScl * mg2oLoopSlw;
        int                 numProjMatches = 0;
        vector<map_point *> vpMatchedMPs;
        bool                bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs);
        if (bCommonRegion)
        {

            bLoopDetectedInKF = true;

            mnLoopNumCoincidences++;
            mpLoopLastCurrentKF->set_erase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw         = gScw;
            mvpLoopMatchedMPs   = vpMatchedMPs;

            mbLoopDetected    = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if (!mbLoopDetected)
            {
                spdlog::debug("Place recognition loop detected with refined Sim3");
            }
        }
        else
        {
            bLoopDetectedInKF = false;

            mnLoopNumNotFound++;
            if (mnLoopNumNotFound >= 2)
            {
                mpLoopLastCurrentKF->set_erase();
                mpLoopMatchedKF->set_erase();
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }
        }
    }

    // Merge candidates
    bool bMergeDetectedInKF = false;
    if (mnMergeNumCoincidences > 0)
    {
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->get_pose() * mpMergeLastCurrentKF->get_pose_inverse()).cast<double>();

        g2o::Sim3           gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3           gScw           = gScl * mg2oMergeSlw;
        int                 numProjMatches = 0;
        vector<map_point *> vpMatchedMPs;
        bool                bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs);
        if (bCommonRegion)
        {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->set_erase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeSlw         = gScw;
            mvpMergeMatchedMPs   = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        }
        else
        {
            mbMergeDetected    = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if (mnMergeNumNotFound >= 2)
            {
                mpMergeLastCurrentKF->set_erase();
                mpMergeMatchedKF->set_erase();
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0;
            }
        }
    }

    if (mbMergeDetected || mbLoopDetected)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        return true;
    }

    // TODO: This is only necessary if we use a minimun score for pick the best candidates
    const vector<key_frame *> vpConnectedKeyFrames = mpCurrentKF->get_vector_covisible_key_frames();

    // Extract candidates from the bag of words
    vector<key_frame *> vpMergeBowCand, vpLoopBowCand;
    if (!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
        // Search in BoW
        mpKeyFrameDB->detect_n_best_candidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand, 3);
    }

    // Check the BoW candidates if the geometric candidate list is empty
    // Loop candidates
    if (!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, mg2oLoopSlw, mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMatchedMPs);
    }
    // Merge candidates
    if (!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF, mg2oMergeSlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMatchedMPs);
    }

    mpKeyFrameDB->add(mpCurrentKF);

    if (mbMergeDetected || mbLoopDetected)
    {
        return true;
    }

    mpCurrentKF->set_erase();
    mpCurrentKF->current_place_recognition = false;

    return false;
}

bool LoopClosing::DetectAndReffineSim3FromLastKF(
    key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs)
{
    set<map_point *> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches    = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if (nNumProjMatches >= nProjMatches)
    {
        Sophus::SE3d                mTwm = pMatchedKF->get_pose_inverse().cast<double>();
        g2o::Sim3                   gSwm(mTwm.unit_quaternion(), mTwm.translation(), 1.0);
        g2o::Sim3                   gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        bool bFixedScale = mbFixScale; // TODO CHECK; Solo para el monocular inertial
        if (mpTracker->mSensor == System::IMU_MONOCULAR && !pCurrentKF->get_map()->get_inertial_ba2())
        {
            bFixedScale = false;
        }
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);

        if (numOptMatches > nProjOptMatches)
        {
            g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(), 1.0);

            vector<map_point *> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->get_map_point_matches().size(), static_cast<map_point *>(NULL));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if (nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<key_frame *>  &vpBowCand,
                                             key_frame                *&pMatchedKF2,
                                             key_frame                *&pLastCurrentKF,
                                             g2o::Sim3                &g2oScw,
                                             int                      &nNumCoincidences,
                                             std::vector<map_point *> &vpMPs,
                                             std::vector<map_point *> &vpMatchedMPs)
{
    int nBoWMatches     = 20;
    int nBoWInliers     = 15;
    int nSim3Inliers    = 20;
    int nProjMatches    = 50;
    int nProjOptMatches = 80;

    set<key_frame *> spConnectedKeyFrames = mpCurrentKF->get_connected_key_frames();

    int nNumCovisibles = 10;

    descriptor_matcher matcherBoW(0.9, true);
    descriptor_matcher matcher(0.75, true);

    // Varibles to select the best numbe
    key_frame                *pBestMatchedKF;
    int                      nBestMatchesReproj   = 0;
    int                      nBestNumCoindicendes = 0;
    g2o::Sim3                g2oBestScw;
    std::vector<map_point *> vpBestMapPoints;
    std::vector<map_point *> vpBestMatchedMapPoints;

    int         numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    for (key_frame *pKFi : vpBowCand)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        // Current KF against KF with covisibles version
        std::vector<key_frame *> vpCovKFi = pKFi->get_best_covisibility_key_frames(nNumCovisibles);
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

        bool bAbortByNearKF = false;
        for (int j = 0; std::cmp_less(j, vpCovKFi.size()); ++j)
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
        int                   numBoWMatches = 0;

        key_frame *pMostBoWMatchesKF  = pKFi;
        int       nMostBoWNumMatches = 0;

        std::vector<map_point *> vpMatchedPoints     = std::vector<map_point *>(mpCurrentKF->get_map_point_matches().size(), static_cast<map_point *>(NULL));
        std::vector<key_frame *>  vpKeyFrameMatchedMP = std::vector<key_frame *>(mpCurrentKF->get_map_point_matches().size(), static_cast<key_frame *>(NULL));

        [[maybe_unused]] int nIndexMostBoWMatchesKF = 0;
        for (int j = 0; std::cmp_less(j, vpCovKFi.size()); ++j)
        {
            if (!vpCovKFi[j] || vpCovKFi[j]->is_bad())
            {
                continue;
            }

            int num = matcherBoW.search_by_vocabulary(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
            if (num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches     = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        for (int j = 0; std::cmp_less(j, vpCovKFi.size()); ++j)
        {
            for (int k = 0; std::cmp_less(k, vvpMatchedMPs[j].size()); ++k)
            {
                map_point *pMPi_j = vvpMatchedMPs[j][k];
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
            bool bFixedScale = mbFixScale;
            if (mpTracker->mSensor == System::IMU_MONOCULAR && !mpCurrentKF->get_map()->get_inertial_ba2())
            {
                bFixedScale = false;
            }

            sim3_solver solver = sim3_solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.set_ransac_parameters(0.99, nBoWInliers, 300); // at least 15 inliers

            bool            bNoMore = false;
            vector<bool>    vbInliers;
            int             nInliers;
            bool            bConverge = false;
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
                vector<key_frame *>  vpKeyFrames;
                for (key_frame *pCovKFi : vpCovKFi)
                {
                    for (map_point *pCovMPij : pCovKFi->get_map_point_matches())
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
                g2o::Sim3     gScw = gScm * gSmw; // Similarity matrix of current from the world position
                Sophus::Sim3f mScw = converter::to_sophus(gScw);

                vector<map_point *> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->get_map_point_matches().size(), static_cast<map_point *>(NULL));
                vector<key_frame *> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->get_map_point_matches().size(), static_cast<key_frame *>(NULL));
                int numProjMatches = matcher.search_by_projection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);

                if (numProjMatches >= nProjMatches)
                {
                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    [[maybe_unused]] bool bFixedScale = mbFixScale;
                    if (mpTracker->mSensor == System::IMU_MONOCULAR && !mpCurrentKF->get_map()->get_inertial_ba2())
                    {
                        bFixedScale = false;
                    }

                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, mHessian7x7, true);

                    if (numOptMatches >= nSim3Inliers)
                    {
                        g2o::Sim3     gSmw(pMostBoWMatchesKF->get_rotation().cast<double>(), pMostBoWMatchesKF->get_translation().cast<double>(), 1.0);
                        g2o::Sim3     gScw = gScm * gSmw; // Similarity matrix of current from the world position
                        Sophus::Sim3f mScw = converter::to_sophus(gScw);

                        vector<map_point *> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->get_map_point_matches().size(), static_cast<map_point *>(NULL));
                        int numProjOptMatches = matcher.search_by_projection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);

                        if (numProjOptMatches >= nProjOptMatches)
                        {
                            int max_x = -1, min_x = 1000000;
                            int max_y = -1, min_y = 1000000;
                            for (map_point *pMPi : vpMatchedMP)
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

                            int nNumKFs = 0;
                            // vpMatchedMPs = vpMatchedMP;
                            // vpMPs = vpMapPoints;
                            //  Check the Sim3 transformation with the current key_frame covisibles
                            vector<key_frame *> vpCurrentCovKFs = mpCurrentKF->get_best_covisibility_key_frames(nNumCovisibles);

                            int j = 0;
                            while (nNumKFs < 3 && std::cmp_less(j, vpCurrentCovKFs.size()))
                            {
                                key_frame           *pKFj = vpCurrentCovKFs[j];
                                Sophus::SE3d        mTjc = (pKFj->get_pose() * mpCurrentKF->get_pose_inverse()).cast<double>();
                                g2o::Sim3           gSjc(mTjc.unit_quaternion(), mTjc.translation(), 1.0);
                                g2o::Sim3           gSjw             = gSjc * gScw;
                                int                 numProjMatches_j = 0;
                                vector<map_point *> vpMatchedMPs_j;
                                bool                bValid = DetectCommonRegionsFromLastKF(pKFj, pMostBoWMatchesKF, gSjw, numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if (bValid)
                                {
                                    Sophus::SE3f                     Tc_w        = mpCurrentKF->get_pose();
                                    Sophus::SE3f                     Tw_cj       = pKFj->get_pose_inverse();
                                    Sophus::SE3f                     Tc_cj       = Tc_w * Tw_cj;
                                    [[maybe_unused]] Eigen::Vector3f vector_dist = Tc_cj.translation();
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
        pLastCurrentKF   = mpCurrentKF;
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
        int                  maxStage = -1;
        [[maybe_unused]] int maxMatched;
        for (int i = 0; std::cmp_less(i, vnStage.size()); ++i)
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

bool LoopClosing::DetectCommonRegionsFromLastKF(
    key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs)
{
    set<map_point *> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    if (nNumProjMatches >= nProjMatches)
    {
        return true;
    }

    return false;
}

int LoopClosing::FindMatchesByProjection(
    key_frame *pCurrentKF, key_frame *pMatchedKFw, g2o::Sim3 &g2oScw, set<map_point *> &spMatchedMPinOrigin, vector<map_point *> &vpMapPoints, vector<map_point *> &vpMatchedMapPoints)
{
    int                nNumCovisibles = 10;
    vector<key_frame *> vpCovKFm       = pMatchedKFw->get_best_covisibility_key_frames(nNumCovisibles);
    int                nInitialCov    = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<key_frame *> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<key_frame *> spCurrentCovisbles = pCurrentKF->get_connected_key_frames();
    if (nInitialCov < nNumCovisibles)
    {
        for (int i = 0; i < nInitialCov; ++i)
        {
            vector<key_frame *> vpKFs     = vpCovKFm[i]->get_best_covisibility_key_frames(nNumCovisibles);
            int                nInserted = 0;
            int                j         = 0;
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
    for (key_frame *pKFi : vpCovKFm)
    {
        for (map_point *pMPij : pKFi->get_map_point_matches())
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

    Sophus::Sim3f      mScw = converter::to_sophus(g2oScw);
    descriptor_matcher matcher(0.9, true);

    vpMatchedMapPoints.resize(pCurrentKF->get_map_point_matches().size(), static_cast<map_point *>(NULL));
    int num_matches = matcher.search_by_projection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

void LoopClosing::CorrectLoop()
{

    // If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA())
    {
        spdlog::info("Stopping global bundle adjustment");
        stop_global_bundle_adjustment();
        spdlog::info("Global bundle adjustment stopped");
    }

    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue();

    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->update_connections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->get_vector_covisible_key_frames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF] = mg2oLoopScw;
    Sophus::SE3f Twc           = mpCurrentKF->get_pose_inverse();
    Sophus::SE3f Tcw           = mpCurrentKF->get_pose();
    g2o::Sim3    g2oScw(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>(), 1.0);
    NonCorrectedSim3[mpCurrentKF] = g2oScw;

    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
    Sophus::SE3d correctedTcw(mg2oLoopScw.rotation(), mg2oLoopScw.translation() / mg2oLoopScw.scale());
    mpCurrentKF->set_pose(correctedTcw.cast<float>());

    map *pLoopMap = mpCurrentKF->get_map();

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->map_update_mutex);

        const bool bImuInit = pLoopMap->is_imu_initialized();

        for (vector<key_frame *>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
        {
            key_frame *pKFi = *vit;

            if (pKFi != mpCurrentKF)
            {
                Sophus::SE3f Tiw = pKFi->get_pose();
                Sophus::SE3d Tic = (Tiw * Twc).cast<double>();
                g2o::Sim3    g2oSic(Tic.unit_quaternion(), Tic.translation(), 1.0);
                g2o::Sim3    g2oCorrectedSiw = g2oSic * mg2oLoopScw;
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
        for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
        {
            key_frame *pKFi            = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)

            vector<map_point *> vpMPsi = pKFi->get_map_point_matches();
            for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
            {
                map_point *pMPi = vpMPsi[iMP];
                if (!pMPi)
                {
                    continue;
                }
                if (pMPi->is_bad())
                {
                    continue;
                }
                if (pMPi->corrected_by_key_frame_id == mpCurrentKF->id)
                {
                    continue;
                }

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw             = pMPi->get_world_position().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw));

                pMPi->set_world_position(eigCorrectedP3Dw.cast<float>());
                pMPi->corrected_by_key_frame_id        = mpCurrentKF->id;
                pMPi->corrected_reference_key_frame_id = pKFi->id;
                pMPi->update_normal_and_depth();
            }

            // Correct velocity according to orientation correction
            if (bImuInit)
            {
                Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * g2oSiw.rotation()).cast<float>();
                pKFi->set_velocity(Rcor * pKFi->get_velocity());
            }

            // Make sure connections are updated
            pKFi->update_connections();
        }
        // TODO Check this index increasement
        mpAtlas->get_current_map()->increase_change_index();

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; std::cmp_less(i, mvpLoopMatchedMPs.size()); i++)
        {
            if (mvpLoopMatchedMPs[i])
            {
                map_point *pLoopMP = mvpLoopMatchedMPs[i];
                map_point *pCurMP  = mpCurrentKF->get_map_point(i);
                if (pCurMP)
                {
                    pCurMP->replace(pLoopMP);
                }
                else
                {
                    mpCurrentKF->add_map_point(pLoopMP, i);
                    pLoopMP->add_observation(mpCurrentKF, i);
                    pLoopMP->compute_distinctive_descriptors();
                }
            }
        }
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);

    // After the map_point fusion, new links in the covisibility graph will appear attaching both sides of the loop
    std::map<key_frame *, set<key_frame *>> LoopConnections;

    for (vector<key_frame *>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
    {
        key_frame          *pKFi                = *vit;
        vector<key_frame *> vpPreviousNeighbors = pKFi->get_vector_covisible_key_frames();

        // Update connections. Detect new links.
        pKFi->update_connections();
        LoopConnections[pKFi] = pKFi->get_connected_key_frames();
        for (vector<key_frame *>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for (vector<key_frame *>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    bool bFixedScale = mbFixScale;
    // TODO CHECK; Solo para el monocular inertial
    if (mpTracker->mSensor == System::IMU_MONOCULAR && !mpCurrentKF->get_map()->get_inertial_ba2())
    {
        bFixedScale = false;
    }

    if (pLoopMap->is_inertial() && pLoopMap->is_imu_initialized())
    {
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }
    else
    {
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale);
    }

    mpAtlas->inform_new_big_change();

    // Add loop edge
    mpLoopMatchedKF->add_loop_edge(mpCurrentKF);
    mpCurrentKF->add_loop_edge(mpLoopMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment (Only if few keyframes, if not it would take too much time)
    if (!pLoopMap->is_imu_initialized() || (pLoopMap->key_frame_count() < 200 && mpAtlas->count_maps() == 1))
    {
        stop_global_bundle_adjustment();
        mbRunningGBA    = true;
        mbFinishedGBA   = false;
        mbStopGBA       = false;
        mnCorrectionGBA = mnNumCorrection;

        _global_ba_thread = std::jthread(&LoopClosing::RunGlobalBundleAdjustment, this, pLoopMap, mpCurrentKF->id);
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    mLastLoopKFid = mpCurrentKF->id; // TODO old varible, it is not use in the new algorithm
}

void LoopClosing::MergeLocal()
{
    int numTemporalKFs = 25; // Temporal KFs in the local window if the map is inertial.

    // Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    key_frame *pNewChild;
    key_frame *pNewParent;

    vector<key_frame *> vpLocalCurrentWindowKFs;
    vector<key_frame *> vpMergeConnectedKFs;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    //  If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA())
    {
        stop_global_bundle_adjustment();
        bRelaunchBA = true;
    }

    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    mpLocalMapper->EmptyQueue();

    // Merge map will become in the new active map with the local window of KFs and MPs from the current map.
    // Later, the elements of the current map will be transform to the new active map reference, in order to keep real time tracking
    map *pCurrentMap = mpCurrentKF->get_map();
    map *pMergeMap   = mpMergeMatchedKF->get_map();

    // Ensure current keyframe is updated
    mpCurrentKF->update_connections();

    // Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    set<key_frame *> spLocalWindowKFs;
    // Get MPs in the welding area from the current map
    set<map_point *> spLocalWindowMPs;
    if (pCurrentMap->is_inertial() && pMergeMap->is_inertial()) // TODO Check the correct initialization
    {
        key_frame *pKFi      = mpCurrentKF;
        int       nInserted = 0;
        while (pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            pKFi = mpCurrentKF->previous_key_frame;
            nInserted++;

            set<map_point *> spMPi = pKFi->get_map_points();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = mpCurrentKF->next_key_frame;
        while (pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            set<map_point *> spMPi = pKFi->get_map_points();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());

            pKFi = mpCurrentKF->next_key_frame;
        }
    }
    else
    {
        spLocalWindowKFs.insert(mpCurrentKF);
    }

    vector<key_frame *> vpCovisibleKFs = mpCurrentKF->get_best_covisibility_key_frames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spLocalWindowKFs.insert(mpCurrentKF);
    const int nMaxTries = 5;
    int       nNumTries = 0;
    while (std::cmp_less(spLocalWindowKFs.size(), numTemporalKFs) && nNumTries < nMaxTries)
    {
        vector<key_frame *> vpNewCovKFs;
        vpNewCovKFs.clear();
        for (key_frame *pKFi : spLocalWindowKFs)
        {
            vector<key_frame *> vpKFiCov = pKFi->get_best_covisibility_key_frames(numTemporalKFs / 2);
            for (key_frame *pKFcov : vpKFiCov)
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

    for (key_frame *pKFi : spLocalWindowKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        set<map_point *> spMPs = pKFi->get_map_points();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }

    set<key_frame *> spMergeConnectedKFs;
    if (pCurrentMap->is_inertial() && pMergeMap->is_inertial()) // TODO Check the correct initialization
    {
        key_frame *pKFi      = mpMergeMatchedKF;
        int       nInserted = 0;
        while (pKFi && nInserted < numTemporalKFs / 2)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->previous_key_frame;
            nInserted++;
        }

        pKFi = mpMergeMatchedKF->next_key_frame;
        while (pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->next_key_frame;
        }
    }
    else
    {
        spMergeConnectedKFs.insert(mpMergeMatchedKF);
    }
    vpCovisibleKFs = mpMergeMatchedKF->get_best_covisibility_key_frames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spMergeConnectedKFs.insert(mpMergeMatchedKF);
    nNumTries = 0;
    while (std::cmp_less(spMergeConnectedKFs.size(), numTemporalKFs) && nNumTries < nMaxTries)
    {
        vector<key_frame *> vpNewCovKFs;
        for (key_frame *pKFi : spMergeConnectedKFs)
        {
            vector<key_frame *> vpKFiCov = pKFi->get_best_covisibility_key_frames(numTemporalKFs / 2);
            for (key_frame *pKFcov : vpKFiCov)
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
    for (key_frame *pKFi : spMergeConnectedKFs)
    {
        set<map_point *> vpMPs = pKFi->get_map_points();
        spMapPointMerge.insert(vpMPs.begin(), vpMPs.end());
    }

    vector<map_point *> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    //
    Sophus::SE3d Twc = mpCurrentKF->get_pose_inverse().cast<double>();
    g2o::Sim3    g2oNonCorrectedSwc(Twc.unit_quaternion(), Twc.translation(), 1.0);
    g2o::Sim3    g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3    g2oCorrectedScw    = mg2oMergeScw; // TODO Check the transformation

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[mpCurrentKF]    = g2oCorrectedScw;
    vNonCorrectedSim3[mpCurrentKF] = g2oNonCorrectedScw;

    for (key_frame *pKFi : spLocalWindowKFs)
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

        if (pKFi != mpCurrentKF)
        {
            Sophus::SE3d Tiw = (pKFi->get_pose()).cast<double>();
            g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
            // Pose without correction
            vNonCorrectedSim3[pKFi] = g2oSiw;

            Sophus::SE3d Tic = Tiw * Twc;
            g2o::Sim3    g2oSic(Tic.unit_quaternion(), Tic.translation(), 1.0);
            g2oCorrectedSiw      = g2oSic * mg2oMergeScw;
            vCorrectedSim3[pKFi] = g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->camera_pose_merge = pKFi->get_pose();

        // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
        double s      = g2oCorrectedSiw.scale();
        pKFi->scale = s;
        Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

        pKFi->camera_pose_merge = correctedTiw.cast<float>();

        if (pCurrentMap->is_imu_initialized())
        {
            Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
            pKFi->velocity_world_merge         = Rcor * pKFi->get_velocity();
        }

        // TODO DEBUG to know which are the KFs that had been moved to the other map
    }

    int numPointsWithCorrection = 0;

    // for(map_point* pMPi : spLocalWindowMPs)
    set<map_point *>::iterator itMP = spLocalWindowMPs.begin();
    while (itMP != spLocalWindowMPs.end())
    {
        map_point *pMPi = *itMP;
        if (!pMPi || pMPi->is_bad())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }

        key_frame *pKFref = pMPi->get_reference_key_frame();
        if (vCorrectedSim3.find(pKFref) == vCorrectedSim3.end())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            numPointsWithCorrection++;
            continue;
        }
        g2o::Sim3 g2oCorrectedSwi    = vCorrectedSim3[pKFref].inverse();
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        Eigen::Vector3d    P3Dw             = pMPi->get_world_position().cast<double>();
        Eigen::Vector3d    eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
        Eigen::Quaterniond Rcor             = g2oCorrectedSwi.rotation() * g2oNonCorrectedSiw.rotation();

        pMPi->merge_position      = eigCorrectedP3Dw.cast<float>();
        pMPi->merge_normal_vector = Rcor.cast<float>() * pMPi->get_normal();

        itMP++;
    }

    {
        unique_lock<mutex> currentLock(pCurrentMap->map_update_mutex); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->map_update_mutex);     // We remove the Kfs and MPs in the merged area from the old map

        for (key_frame *pKFi : spLocalWindowKFs)
        {
            if (!pKFi || pKFi->is_bad())
            {
                continue;
            }

            pKFi->camera_pose_before_merge = pKFi->get_pose();
            pKFi->world_pose_before_merge = pKFi->get_pose_inverse();
            pKFi->set_pose(pKFi->camera_pose_merge);

            // Make sure connections are updated
            pKFi->update_map(pMergeMap);
            pKFi->merge_corrected_key_frame_id = mpCurrentKF->id;
            pMergeMap->add_key_frame(pKFi);
            pCurrentMap->erase_key_frame(pKFi);

            if (pCurrentMap->is_imu_initialized())
            {
                pKFi->set_velocity(pKFi->velocity_world_merge);
            }
        }

        for (map_point *pMPi : spLocalWindowMPs)
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

        mpAtlas->change_map(pMergeMap);
        mpAtlas->set_map_bad(pCurrentMap);
        pMergeMap->increase_change_index();
        // TODO for debug
        pMergeMap->change_id(pCurrentMap->get_id());
    }

    // Rebuild the essential graph in the local window
    pCurrentMap->get_origin_key_frame()->set_first_connection(false);
    pNewChild  = mpCurrentKF->get_parent(); // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF;              // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpCurrentKF->change_parent(mpMergeMatchedKF);
    while (pNewChild)
    {
        pNewChild->erase_child(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        key_frame *pOldParent = pNewChild->get_parent();

        pNewChild->change_parent(pNewParent);

        pNewParent = pNewChild;
        pNewChild  = pOldParent;
    }

    // Update the connections between the local window
    mpMergeMatchedKF->update_connections();

    vpMergeConnectedKFs = mpMergeMatchedKF->get_vector_covisible_key_frames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(vCorrectedSim3, vpCheckFuseMapPoint);

    // Update connectivity
    for (key_frame *pKFi : spLocalWindowKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        pKFi->update_connections();
    }
    for (key_frame *pKFi : spMergeConnectedKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        pKFi->update_connections();
    }

    bool bStop = false;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    if (mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD)
    {
        Optimizer::MergeInertialBA(mpCurrentKF, mpMergeMatchedKF, &bStop, pCurrentMap, vCorrectedSim3);
    }
    else
    {
        Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs, vpMergeConnectedKFs, &bStop);
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    // Update the non critical area from the current map to the merged map
    vector<key_frame *>  vpCurrentMapKFs = pCurrentMap->get_all_key_frames();
    vector<map_point *> vpCurrentMapMPs = pCurrentMap->get_all_map_points();

    if (vpCurrentMapKFs.size() == 0)
    {
    }
    else
    {
        if (mpTracker->mSensor == System::MONOCULAR)
        {
            unique_lock<mutex> currentLock(pCurrentMap->map_update_mutex); // We update the current map with the Merge information

            for (key_frame *pKFi : vpCurrentMapKFs)
            {
                if (!pKFi || pKFi->is_bad() || pKFi->get_map() != pCurrentMap)
                {
                    continue;
                }

                g2o::Sim3 g2oCorrectedSiw;

                Sophus::SE3d Tiw = (pKFi->get_pose()).cast<double>();
                g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
                // Pose without correction
                vNonCorrectedSim3[pKFi] = g2oSiw;

                Sophus::SE3d Tic = Tiw * Twc;
                g2o::Sim3    g2oSim(Tic.unit_quaternion(), Tic.translation(), 1.0);
                g2oCorrectedSiw      = g2oSim * mg2oMergeScw;
                vCorrectedSim3[pKFi] = g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                double s = g2oCorrectedSiw.scale();

                pKFi->scale = s;

                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

                pKFi->camera_pose_before_merge = pKFi->get_pose();
                pKFi->world_pose_before_merge = pKFi->get_pose_inverse();

                pKFi->set_pose(correctedTiw.cast<float>());

                if (pCurrentMap->is_imu_initialized())
                {
                    Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
                    pKFi->set_velocity(Rcor * pKFi->get_velocity()); // TODO: should add here scale s
                }
            }
            for (map_point *pMPi : vpCurrentMapMPs)
            {
                if (!pMPi || pMPi->is_bad() || pMPi->get_map() != pCurrentMap)
                {
                    continue;
                }

                key_frame *pKFref             = pMPi->get_reference_key_frame();
                g2o::Sim3 g2oCorrectedSwi    = vCorrectedSim3[pKFref].inverse();
                g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw             = pMPi->get_world_position().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
                pMPi->set_world_position(eigCorrectedP3Dw.cast<float>());

                pMPi->update_normal_and_depth();
            }
        }

        mpLocalMapper->RequestStop();
        // Wait until Local Mapping has effectively stopped
        while (!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }

        // Optimize graph (and update the loop position for each element form the begining to the end)
        if (mpTracker->mSensor != System::MONOCULAR)
        {
            Optimizer::OptimizeEssentialGraph(mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }

        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->map_update_mutex); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->map_update_mutex);     // We remove the Kfs and MPs in the merged area from the old map

            for (key_frame *pKFi : vpCurrentMapKFs)
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

            for (map_point *pMPi : vpCurrentMapMPs)
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

    mpLocalMapper->Release();

    if (bRelaunchBA && (!pCurrentMap->is_imu_initialized() || (pCurrentMap->key_frame_count() < 200 && mpAtlas->count_maps() == 1)))
    {
        // Launch a new thread to perform Global Bundle Adjustment
        stop_global_bundle_adjustment();
        mbRunningGBA      = true;
        mbFinishedGBA     = false;
        mbStopGBA         = false;
        _global_ba_thread = std::jthread(&LoopClosing::RunGlobalBundleAdjustment, this, pMergeMap, mpCurrentKF->id);
    }

    mpMergeMatchedKF->add_merge_edge(mpCurrentKF);
    mpCurrentKF->add_merge_edge(mpMergeMatchedKF);

    pCurrentMap->increase_change_index();
    pMergeMap->increase_change_index();
}

void LoopClosing::MergeLocal2()
{

    [[maybe_unused]] int numTemporalKFs = 11; // TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    // Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    key_frame *pNewChild;
    key_frame *pNewParent;

    vector<key_frame *> vpLocalCurrentWindowKFs;
    vector<key_frame *> vpMergeConnectedKFs;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    // NonCorrectedSim3[mpCurrentKF]=mg2oLoopScw;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    [[maybe_unused]] bool bRelaunchBA = false;

    //  If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA())
    {
        stop_global_bundle_adjustment();
        bRelaunchBA = true;
    }

    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    map *pCurrentMap = mpCurrentKF->get_map();
    map *pMergeMap   = mpMergeMatchedKF->get_map();

    {
        float        s_on = mSold_new.scale();
        Sophus::SE3f T_on(mSold_new.rotation().cast<float>(), mSold_new.translation().cast<float>());

        unique_lock<mutex> lock(mpAtlas->get_current_map()->map_update_mutex);

        mpLocalMapper->EmptyQueue();

        bool bScaleVel = false;
        if (s_on != 1)
        {
            bScaleVel = true;
        }
        mpAtlas->get_current_map()->apply_scaled_rotation(T_on, s_on, bScaleVel);
        mpTracker->UpdateFrameIMU(s_on, mpCurrentKF->get_imu_bias(), mpTracker->GetLastKeyFrame());
    }

    const int numKFnew = pCurrentMap->key_frame_count();

    if ((mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD) && !pCurrentMap->get_inertial_ba2())
    {
        // Map is not completly initialized
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.;
        ba << 0., 0., 0.;
        Optimizer::InertialOptimization(pCurrentMap, bg, ba);
        IMU::Bias          b(ba[0], ba[1], ba[2], bg[0], bg[1], bg[2]);
        unique_lock<mutex> lock(mpAtlas->get_current_map()->map_update_mutex);
        mpTracker->UpdateFrameIMU(1.0f, b, mpTracker->GetLastKeyFrame());

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

        vector<key_frame *>  vpMergeMapKFs = pMergeMap->get_all_key_frames();
        vector<map_point *> vpMergeMapMPs = pMergeMap->get_all_map_points();

        for (key_frame *pKFi : vpMergeMapKFs)
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

        for (map_point *pMPi : vpMergeMapMPs)
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
        vector<key_frame *> vpKFs = pCurrentMap->get_all_key_frames();
        for (key_frame *pKFi : vpKFs)
        {
            Sophus::SE3d Tiw = (pKFi->get_pose()).cast<double>();
            g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
            NonCorrectedSim3[pKFi] = g2oSiw;
        }
    }

    // Critical zone

    pMergeMap->get_origin_key_frame()->set_first_connection(false);
    pNewChild  = mpMergeMatchedKF->get_parent(); // Old parent, it will be the new child of this KF
    pNewParent = mpMergeMatchedKF;              // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpMergeMatchedKF->change_parent(mpCurrentKF);
    while (pNewChild)
    {
        pNewChild->erase_child(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        key_frame *pOldParent = pNewChild->get_parent();
        pNewChild->change_parent(pNewParent);
        pNewParent = pNewChild;
        pNewChild  = pOldParent;
    }

    vector<map_point *> vpCheckFuseMapPoint; // map_point vector from current map to allow to fuse duplicated points with the old map (merge)
    vector<key_frame *>  vpCurrentConnectedKFs;

    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vector<key_frame *> aux = mpMergeMatchedKF->get_vector_covisible_key_frames();
    mvpMergeConnectedKFs.insert(mvpMergeConnectedKFs.end(), aux.begin(), aux.end());
    if (mvpMergeConnectedKFs.size() > 6)
    {
        mvpMergeConnectedKFs.erase(mvpMergeConnectedKFs.begin() + 6, mvpMergeConnectedKFs.end());
    }

    mpCurrentKF->update_connections();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    aux = mpCurrentKF->get_vector_covisible_key_frames();
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end());
    if (vpCurrentConnectedKFs.size() > 6)
    {
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin() + 6, vpCurrentConnectedKFs.end());
    }

    set<map_point *> spMapPointMerge;
    for (key_frame *pKFi : mvpMergeConnectedKFs)
    {
        set<map_point *> vpMPs = pKFi->get_map_points();
        spMapPointMerge.insert(vpMPs.begin(), vpMPs.end());
        if (spMapPointMerge.size() > 1000)
        {
            break;
        }
    }

    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    SearchAndFuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint);

    for (key_frame *pKFi : vpCurrentConnectedKFs)
    {
        if (!pKFi || pKFi->is_bad())
        {
            continue;
        }

        pKFi->update_connections();
    }
    for (key_frame *pKFi : mvpMergeConnectedKFs)
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
        mpLocalMapper->Release();
        return;
    }

    // Perform BA
    bool      bStopFlag = false;
    key_frame *pCurrKF   = mpTracker->GetLastKeyFrame();
    Optimizer::MergeInertialBA(pCurrKF, mpMergeMatchedKF, &bStopFlag, pCurrentMap, CorrectedSim3);

    // Release Local Mapping.
    mpLocalMapper->Release();

    return;
}

void LoopClosing::CheckObservations(set<key_frame *> &spKFsMap1, set<key_frame *> &spKFsMap2)
{
    spdlog::debug("----------------------");
    for (key_frame *pKFi1 : spKFsMap1)
    {
        std::map<key_frame *, int> mMatchedMP;
        set<map_point *>          spMPs = pKFi1->get_map_points();

        for (map_point *pMPij : spMPs)
        {
            if (!pMPij || pMPij->is_bad())
            {
                continue;
            }

            std::map<key_frame *, tuple<int, int>> mMPijObs = pMPij->get_observations();
            for (key_frame *pKFi2 : spKFsMap2)
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

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<map_point *> &vpMapPoints)
{
    descriptor_matcher matcher(0.8);

    int total_replaces = 0;

    for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++)
    {
        int       num_replaces = 0;
        key_frame *pKFi         = mit->first;
        map      *pMap         = pKFi->get_map();

        g2o::Sim3     g2oScw = mit->second;
        Sophus::Sim3f Scw    = converter::to_sophus(g2oScw);

        vector<map_point *>  vpReplacePoints(vpMapPoints.size(), static_cast<map_point *>(NULL));
        [[maybe_unused]] int numFused = matcher.Fuse(pKFi, Scw, vpMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->map_update_mutex);
        const int          nLP = vpMapPoints.size();
        for (int i = 0; i < nLP; i++)
        {
            map_point *pRep = vpReplacePoints[i];
            if (pRep)
            {

                num_replaces += 1;
                pRep->replace(vpMapPoints[i]);
            }
        }

        total_replaces += num_replaces;
    }
}

void LoopClosing::SearchAndFuse(const vector<key_frame *> &vConectedKFs, vector<map_point *> &vpMapPoints)
{
    descriptor_matcher matcher(0.8);

    [[maybe_unused]] int total_replaces = 0;

    for (auto mit = vConectedKFs.begin(), mend = vConectedKFs.end(); mit != mend; mit++)
    {
        int           num_replaces = 0;
        key_frame     *pKF          = (*mit);
        map          *pMap         = pKF->get_map();
        Sophus::SE3f  Tcw          = pKF->get_pose();
        Sophus::Sim3f Scw(Tcw.unit_quaternion(), Tcw.translation());
        Scw.setScale(1.f);
        vector<map_point *> vpReplacePoints(vpMapPoints.size(), static_cast<map_point *>(NULL));
        matcher.Fuse(pKF, Scw, vpMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->map_update_mutex);
        const int          nLP = vpMapPoints.size();
        for (int i = 0; i < nLP; i++)
        {
            map_point *pRep = vpReplacePoints[i];
            if (pRep)
            {
                num_replaces += 1;
                pRep->replace(vpMapPoints[i]);
            }
        }
    }
}

void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
            {
                break;
            }
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset              = pMap;
    }

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetActiveMapRequested)
            {
                break;
            }
        }
        usleep(3000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested)
    {
        spdlog::debug("Loop closer reset requested");
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid             = 0; // TODO old variable, it is not use in the new algorithm
        mbResetRequested          = false;
        mbResetActiveMapRequested = false;
    }
    else if (mbResetActiveMapRequested)
    {

        for (list<key_frame *>::const_iterator it = mlpLoopKeyFrameQueue.begin(); it != mlpLoopKeyFrameQueue.end();)
        {
            key_frame *pKFi = *it;
            if (pKFi->get_map() == mpMapToReset)
            {
                it = mlpLoopKeyFrameQueue.erase(it);
            }
            else
            {
                ++it;
            }
        }

        mLastLoopKFid             = mpAtlas->get_last_init_kf_id(); // TODO old variable, it is not use in the new algorithm
        mbResetActiveMapRequested = false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(map *pActiveMap, unsigned long nLoopKF)
{
    const processing_statistics::backend_timer statistics_timer(_statistics, processing_statistics::backend_stage::GLOBAL_BUNDLE_ADJUSTMENT);
    spdlog::debug("Starting Global Bundle Adjustment");

    const bool bImuInit = pActiveMap->is_imu_initialized();

    if (!bImuInit)
    {
        Optimizer::GlobalBundleAdjustemnt(pActiveMap, 10, &mbStopGBA, nLoopKF, false);
    }
    else
    {
        Optimizer::FullInertialBA(pActiveMap, 7, false, nLoopKF, &mbStopGBA);
    }

    int idx = mnFullBAIdx;

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if (idx != mnFullBAIdx)
        {
            return;
        }

        if (!bImuInit && pActiveMap->is_imu_initialized())
        {
            return;
        }

        if (!mbStopGBA)
        {
            spdlog::debug("Global Bundle Adjustment finished");
            spdlog::debug("Updating map ...");

            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->map_update_mutex);

            //  Correct keyframes starting at map first keyframe
            list<key_frame *> lpKFtoCheck(pActiveMap->key_frame_origins.begin(), pActiveMap->key_frame_origins.end());

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

                    if (pChild->global_ba_key_frame_id != nLoopKF)
                    {
                        Sophus::SE3f Tchildc = pChild->get_pose() * Twc;
                        pChild->camera_pose_global_ba      = Tchildc * pKF->camera_pose_global_ba; //*Tcorc*pKF->camera_pose_global_ba;

                        Sophus::SO3f Rcor = pChild->camera_pose_global_ba.so3().inverse() * pChild->get_pose().so3();
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
            const vector<map_point *> vpMPs = pActiveMap->get_all_map_points();

            for (size_t i = 0; std::cmp_less(i, vpMPs.size()); i++)
            {
                map_point *pMP = vpMPs[i];

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
                    key_frame *pRefKF = pMP->get_reference_key_frame();

                    if (pRefKF->global_ba_key_frame_id != nLoopKF)
                    {
                        continue;
                    }

                    // Map to non-corrected camera
                    // cv::Mat Rcw = pRefKF->camera_pose_before_global_ba.rowRange(0,3).colRange(0,3);
                    // cv::Mat tcw = pRefKF->camera_pose_before_global_ba.rowRange(0,3).col(3);
                    Eigen::Vector3f Xc = pRefKF->camera_pose_before_global_ba * pMP->get_world_position();

                    // Backproject using corrected camera
                    pMP->set_world_position(pRefKF->get_pose_inverse() * Xc);
                }
            }

            pActiveMap->inform_new_big_change();
            pActiveMap->increase_change_index();

            // TODO Check this update

            mpLocalMapper->Release();

            spdlog::debug("Map updated!");
        }

        mbFinishedGBA = true;
        mbRunningGBA  = false;
    }
}

void LoopClosing::stop_global_bundle_adjustment()
{
    {
        const std::lock_guard lock(mMutexGBA);
        mbStopGBA = true;
        ++mnFullBAIdx;
    }
    if (_global_ba_thread.joinable())
    {
        _global_ba_thread.request_stop();
        _global_ba_thread.join();
    }
    const std::lock_guard lock(mMutexGBA);
    mbRunningGBA  = false;
    mbFinishedGBA = true;
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} // namespace ORB_SLAM3
