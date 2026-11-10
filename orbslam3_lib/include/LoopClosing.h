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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "key_frame.h"
#include "local_mapping.h"
#include "Tracking.h"
#include "atlas.h"
#include "key_frame_database.h"
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

class LoopClosing
{
  public:
    typedef pair<set<key_frame *>, int>                                                                                              ConsistentGroup;
    typedef std::map<key_frame *, g2o::Sim3, std::less<key_frame *>, Eigen::aligned_allocator<std::pair<key_frame *const, g2o::Sim3>>> KeyFrameAndPose;

  public:
    LoopClosing(processing_statistics &statistics, atlas *pAtlas, key_frame_database *pDB, place_vocabulary *pVoc, const bool bFixScale, const bool bActiveLC);
    ~LoopClosing();

    void SetTracker(Tracking *pTracker);

    void SetLocalMapper(local_mapping *pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(key_frame *pKF);

    void RequestReset();
    void RequestResetActiveMap(map *pMap);

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(map *pActiveMap, unsigned long nLoopKF);

    bool isRunningGBA()
    {
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA()
    {
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }

    void RequestFinish();

    void stop_global_bundle_adjustment();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  protected:
    processing_statistics &_statistics;

    bool CheckNewKeyFrames();

    // Methods to implement the new place recognition algorithm
    bool NewDetectCommonRegions();
    bool
    DetectAndReffineSim3FromLastKF(key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs);
    bool DetectCommonRegionsFromBoW(std::vector<key_frame *>  &vpBowCand,
                                    key_frame                *&pMatchedKF,
                                    key_frame                *&pLastCurrentKF,
                                    g2o::Sim3                &g2oScw,
                                    int                      &nNumCoincidences,
                                    std::vector<map_point *> &vpMPs,
                                    std::vector<map_point *> &vpMatchedMPs);
    bool
    DetectCommonRegionsFromLastKF(key_frame *pCurrentKF, key_frame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, std::vector<map_point *> &vpMPs, std::vector<map_point *> &vpMatchedMPs);
    int FindMatchesByProjection(
        key_frame *pCurrentKF, key_frame *pMatchedKFw, g2o::Sim3 &g2oScw, set<map_point *> &spMatchedMPinOrigin, vector<map_point *> &vpMapPoints, vector<map_point *> &vpMatchedMapPoints);

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<map_point *> &vpMapPoints);
    void SearchAndFuse(const vector<key_frame *> &vConectedKFs, vector<map_point *> &vpMapPoints);

    void CorrectLoop();

    void MergeLocal();
    void MergeLocal2();

    void CheckObservations(set<key_frame *> &spKFsMap1, set<key_frame *> &spKFsMap2);

    void       ResetIfRequested();
    bool       mbResetRequested;
    bool       mbResetActiveMapRequested;
    map       *mpMapToReset;
    std::mutex mMutexReset;

    bool       CheckFinish();
    void       SetFinish();
    bool       mbFinishRequested;
    bool       mbFinished;
    std::mutex mMutexFinish;

    atlas    *mpAtlas;
    Tracking *mpTracker;

    key_frame_database *mpKeyFrameDB;
    place_vocabulary   *place_vocabulary_ptr;

    local_mapping *mpLocalMapper;

    std::list<key_frame *> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    key_frame                    *mpCurrentKF;
    key_frame                    *mpLastCurrentKF;
    key_frame                    *mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<key_frame *>      mvpEnoughConsistentCandidates;
    std::vector<key_frame *>      mvpCurrentConnectedKFs;
    std::vector<map_point *>     mvpCurrentMatchedPoints;
    std::vector<map_point *>     mvpLoopMapPoints;
    cv::Mat                      mScw;
    g2o::Sim3                    mg2oScw;

    //-------
    map *mpLastMap;

    bool                     mbLoopDetected;
    int                      mnLoopNumCoincidences;
    int                      mnLoopNumNotFound;
    key_frame                *mpLoopLastCurrentKF;
    g2o::Sim3                mg2oLoopSlw;
    g2o::Sim3                mg2oLoopScw;
    key_frame                *mpLoopMatchedKF;
    std::vector<map_point *> mvpLoopMPs;
    std::vector<map_point *> mvpLoopMatchedMPs;
    bool                     mbMergeDetected;
    int                      mnMergeNumCoincidences;
    int                      mnMergeNumNotFound;
    key_frame                *mpMergeLastCurrentKF;
    g2o::Sim3                mg2oMergeSlw;
    g2o::Sim3                mg2oMergeSmw;
    g2o::Sim3                mg2oMergeScw;
    key_frame                *mpMergeMatchedKF;
    std::vector<map_point *> mvpMergeMPs;
    std::vector<map_point *> mvpMergeMatchedMPs;
    std::vector<key_frame *>  mvpMergeConnectedKFs;

    g2o::Sim3 mSold_new;
    //-------

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool         mbRunningGBA;
    bool         mbFinishedGBA;
    bool         mbStopGBA;
    std::mutex   mMutexGBA;
    std::jthread _global_ba_thread;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;

    int mnFullBAIdx{};

    vector<double> vdPR_CurrentTime;
    vector<double> vdPR_MatchedTime;
    vector<int>    vnPR_TypeRecogn;

    // DEBUG
    string mstrFolderSubTraj;
    int    mnNumCorrection;
    int    mnCorrectionGBA;

    // To (de)activate LC
    bool mbActiveLC = true;

#ifdef REGISTER_LOOP
    string mstrFolderLoop;
#endif
};

} // namespace ORB_SLAM3

#endif // LOOPCLOSING_H
