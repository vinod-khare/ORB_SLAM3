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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "key_frame.h"
#include "LoopClosing.h"
#include "settings.h"
#include "Tracking.h"
#include "atlas.h"
#include "key_frame_database.h"
#include "processing_statistics.h"

#include <mutex>

namespace ORB_SLAM3
{

class System;
class Tracking;
class LoopClosing;

class LocalMapping
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LocalMapping(processing_statistics &statistics, System *pSys, atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName = std::string());

    void SetLoopCloser(LoopClosing *pLoopCloser);

    void SetTracker(Tracking *pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(key_frame *pKF);
    void EmptyQueue();

    // Thread Synch
    void RequestStop();
    void RequestReset();
    void RequestResetActiveMap(map *pMap);
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue()
    {
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    bool      IsInitializing();
    double    GetCurrKFTime();
    key_frame *GetCurrKF();

    std::mutex mMutexImuInit;

    Eigen::MatrixXd mcovInertial;
    Eigen::Matrix3d mRwg;
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double          mScale;
    double          mInitTime;
    double          mCostTime;

    unsigned int mInitSect;
    unsigned int mIdxInit;
    unsigned int mnKFs;
    double       mFirstTs;
    int          mnMatchesInliers;

    // For debugging (erase in normal mode)
    int    mInitFr;
    int    mIdxIteration;
    string strSequence;

    bool mbNotBA1;
    bool mbNotBA2;
    bool mbBadImu;

    bool mbWriteStats;

    // not consider far points (clouds)
    bool  mbFarPoints;
    float mThFarPoints;

  protected:
    processing_statistics &_statistics;

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();

    System *mpSystem;

    bool mbMonocular;
    bool mbInertial;

    void       ResetIfRequested();
    bool       mbResetRequested;
    bool       mbResetRequestedActiveMap;
    map       *mpMapToReset;
    std::mutex mMutexReset;

    bool       CheckFinish();
    void       SetFinish();
    bool       mbFinishRequested;
    bool       mbFinished;
    std::mutex mMutexFinish;

    atlas *mpAtlas;

    LoopClosing *mpLoopCloser;
    Tracking    *mpTracker;

    std::list<key_frame *> mlNewKeyFrames;

    key_frame *mpCurrentKeyFrame;

    std::list<map_point *> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool       mbStopped;
    bool       mbStopRequested;
    bool       mbNotStop;
    std::mutex mMutexStop;

    bool       mbAcceptKeyFrames;
    std::mutex mMutexAccept;

    void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
    void ScaleRefinement();

    bool bInitializing;

    Eigen::MatrixXd infoInertial;
    int             mNumLM;
    int             mNumKFCulling;

    float mTinit;

    int countRefinement;

    // DEBUG
    ofstream f_lm;
};

} // namespace ORB_SLAM3

#endif // LOCALMAPPING_H
