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

#include "KeyFrame.h"

#include "ImuTypes.h"
#include "converter.h"

#include <mutex>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

namespace
{
std::vector<keypoint> keypoints_with_color(frame &frame, const std::vector<keypoint> &keys, int index_offset)
{
    auto colored_keys = keys;
    for (std::size_t index = 0; index < colored_keys.size(); ++index)
    {
        const int frame_index         = index_offset + static_cast<int>(index);
        colored_keys[index].color     = frame.keypoint_color(frame_index);
        const auto &image             = frame.is_fisheye_stereo && frame_index >= static_cast<int>(frame.keypoints.size()) ? frame.color_image_right : frame.color_image;
        colored_keys[index].has_color = !image.empty();
    }
    return colored_keys;
}
} // namespace

long unsigned int KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(frame &F, map *pMap, key_frame_database *pKFDB, const camera_vector &cameras)
    : bImu(pMap->is_imu_initialized()), mnFrameId(F.id), mTimeStamp(F.timestamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(F.grid_element_width_inverse),
      mfGridElementHeightInv(F.grid_element_height_inverse), mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnNumberOfOpt(0), mnLoopQuery(0),
      mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0), mbCurrentPlaceRecognition(false), mnBAGlobalForKF(0),
      mnMergeCorrectedForKF(0), mnBALocalForMerge(0), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy), mbf(F.baseline_fx), mb(F.baseline), mThDepth(F.depth_threshold),
      mDistCoef(F.distortion_coefficients), N(F.feature_count), mvKeys(keypoints_with_color(F, F.keypoints, 0)), mvuRight(F.right_coordinates), mvDepth(F.depths), mBowVec(F.bow_vector),
      mFeatVec(F.feature_vector), mnScaleLevels(F.scale_levels), mfScaleFactor(F.scale_factor), mfLogScaleFactor(F.log_scale_factor), mvScaleFactors(F.scale_factors),
      mvLevelSigma2(F.level_sigma_squared), mvInvLevelSigma2(F.inverse_level_sigma_squared), mnMinX(F.min_x), mnMinY(F.min_y), mnMaxX(F.max_x), mnMaxY(F.max_y), mPrevKF(nullptr),
      mNextKF(nullptr), mpImuPreintegrated(F.imu_preintegrated), mImuCalib(F.imu_calibration), mNameFile(F.name_file), mnDataset(F.dataset), mbHasVelocity(false),
      mTlr(F.get_relative_pose_tlr()), mTrl(F.get_relative_pose_trl()), mvpMapPoints(F.map_points), mpKeyFrameDB(pKFDB), place_vocabulary_ptr(F.place_vocabulary_ptr), mbFirstConnection(true),
      mpParent(nullptr), mbNotErase(false), mbToBeErased(false), mbBad(false), mHalfBaseline(F.baseline / 2), mpMap(pMap), mK_(F.eigen_calibration_matrix), cameras(cameras),
      mvLeftToRightMatch(F.left_to_right_matches), mvRightToLeftMatch(F.right_to_left_matches), mvKeysRight(keypoints_with_color(F, F.right_keypoints, static_cast<int>(F.keypoints.size())))
{
    mnId = nNextId++;
    mGrid.resize(mnGridCols);
    if (is_fisheye_stereo(cameras))
    {
        mGridRight.resize(mnGridCols);
    }
    for (int i = 0; i < mnGridCols; i++)
    {
        mGrid[i].resize(mnGridRows);
        if (is_fisheye_stereo(cameras))
        {
            mGridRight[i].resize(mnGridRows);
        }
        for (int j = 0; j < mnGridRows; j++)
        {
            mGrid[i][j] = F.grid[i][j];
            if (is_fisheye_stereo(cameras))
            {
                mGridRight[i][j] = F.right_grid[i][j];
            }
        }
    }

    if (!F.has_velocity())
    {
        mVw.setZero();
        mbHasVelocity = false;
    }
    else
    {
        mVw           = F.get_velocity();
        mbHasVelocity = true;
    }

    mImuBias = F.imu_bias;
    SetPose(F.get_pose());

    mnOriginMapId = pMap->get_id();
}

void KeyFrame::ComputeBoW()
{
    if (mBowVec.empty() || mFeatVec.empty())
    {
        auto       vCurrentDesc      = descriptor_vector(mvKeys);
        const auto right_descriptors = descriptor_vector(mvKeysRight);
        vCurrentDesc.insert(vCurrentDesc.end(), right_descriptors.begin(), right_descriptors.end());
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        place_vocabulary_ptr->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void KeyFrame::SetPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexPose);

    mTcw = Tcw;
    mRcw = mTcw.rotationMatrix();
    mTwc = mTcw.inverse();
    mRwc = mTwc.rotationMatrix();

    if (mImuCalib.mbIsSet) // TODO Use a flag instead of the OpenCV matrix
    {
        mOwb = mRwc * mImuCalib.mTcb.translation() + mTwc.translation();
    }
}

void KeyFrame::SetVelocity(const Eigen::Vector3f &Vw)
{
    unique_lock<mutex> lock(mMutexPose);
    mVw           = Vw;
    mbHasVelocity = true;
}

Sophus::SE3f KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw;
}

Sophus::SE3f KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc;
}

Eigen::Vector3f KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc.translation();
}

Eigen::Vector3f KeyFrame::GetImuPosition()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOwb;
}

Eigen::Matrix3f KeyFrame::GetImuRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mTwc * mImuCalib.mTcb).rotationMatrix();
}

Sophus::SE3f KeyFrame::GetImuPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc * mImuCalib.mTcb;
}

Eigen::Matrix3f KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mRcw;
}

Eigen::Vector3f KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw.translation();
}

Eigen::Vector3f KeyFrame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexPose);
    return mVw;
}

bool KeyFrame::isVelocitySet()
{
    unique_lock<mutex> lock(mMutexPose);
    return mbHasVelocity;
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (!mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights[pKF] = weight;
        }
        else if (mConnectedKeyFrameWeights[pKF] != weight)
        {
            mConnectedKeyFrameWeights[pKF] = weight;
        }
        else
        {
            return;
        }
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex>            lock(mMutexConnections);
    vector<pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for (std::map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
    {
        vPairs.push_back(make_pair(mit->second, mit->first));
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame *> lKFs;
    list<int>        lWs;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        if (!vPairs[i].second->isBad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights             = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame *> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame *>    s;
    for (std::map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
    {
        s.insert(mit->first);
    }
    return s;
}

vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
    {
        return mvpOrderedConnectedKeyFrames;
    }
    else
    {
        return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
    }
}

vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty())
    {
        return vector<KeyFrame *>();
    }

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::weightComp);

    if (it == mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return vector<KeyFrame *>();
    }
    else
    {
        int n = it - mvOrderedWeights.begin();
        return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
    {
        return mConnectedKeyFrameWeights[pKF];
    }
    else
    {
        return 0;
    }
}

int KeyFrame::GetNumberMPs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    int                numberMPs = 0;
    for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
    {
        if (!mvpMapPoints[i])
        {
            continue;
        }
        numberMPs++;
    }
    return numberMPs;
}

void KeyFrame::AddMapPoint(map_point *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const int &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = static_cast<map_point *>(NULL);
}

void KeyFrame::EraseMapPointMatch(map_point *pMP)
{
    const auto indexes   = pMP->get_index_in_key_frame(this);
    const int  leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    if (leftIndex != -1)
    {
        mvpMapPoints[leftIndex] = static_cast<map_point *>(NULL);
    }
    if (rightIndex != -1)
    {
        mvpMapPoints[rightIndex] = static_cast<map_point *>(NULL);
    }
}

void KeyFrame::ReplaceMapPointMatch(const int &idx, map_point *pMP) { mvpMapPoints[idx] = pMP; }

set<map_point *> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<map_point *>   s;
    for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
    {
        if (!mvpMapPoints[i])
        {
            continue;
        }
        map_point *pMP = mvpMapPoints[i];
        if (!pMP->is_bad())
        {
            s.insert(pMP);
        }
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int        nPoints   = 0;
    const bool bCheckObs = minObs > 0;
    for (int i = 0; i < N; i++)
    {
        map_point *pMP = mvpMapPoints[i];
        // TODO: reduce nesting
        if (pMP)
        {
            if (!pMP->is_bad())
            {
                if (bCheckObs)
                {
                    if (mvpMapPoints[i]->get_observation_count() >= minObs)
                    {
                        nPoints++;
                    }
                }
                else
                {
                    nPoints++;
                }
            }
        }
    }

    return nPoints;
}

vector<map_point *> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

map_point *KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections(bool upParent)
{
    std::map<KeyFrame *, int> KFcounter;

    vector<map_point *> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    // For all map points in keyframe check in which other keyframes are they seen
    // Increase counter for those keyframes
    for (vector<map_point *>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
    {
        map_point *pMP = *vit;

        if (!pMP)
        {
            continue;
        }

        if (pMP->is_bad())
        {
            continue;
        }

        std::map<KeyFrame *, tuple<int, int>> observations = pMP->get_observations();

        for (std::map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            if (mit->first->mnId == mnId || mit->first->isBad() || mit->first->GetMap() != mpMap)
            {
                continue;
            }
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if (KFcounter.empty())
    {
        return;
    }

    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum counter
    int       nmax   = 0;
    KeyFrame *pKFmax = NULL;
    int       th     = 15;

    vector<pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(KFcounter.size());
    if (!upParent)
    {
        spdlog::debug("Updating connections for keyframe {}", mnId);
    }

    // TODO: use modern iterators
    for (std::map<KeyFrame *, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
    {
        if (!upParent)
        {
            spdlog::trace("Connected keyframe {} has {} matches", mit->first->mnId, mit->second);
        }
        if (mit->second > nmax)
        {
            nmax   = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th)
        {
            vPairs.push_back(make_pair(mit->second, mit->first));
            (mit->first)->AddConnection(this, mit->second);
        }
    }

    if (vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->AddConnection(this, nmax);
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame *> lKFs;
    list<int>        lWs;
    for (size_t i = 0; i < vPairs.size(); i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights    = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
        mvOrderedWeights             = vector<int>(lWs.begin(), lWs.end());

        if (mbFirstConnection && mnId != mpMap->get_initial_key_frame_id())
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }
    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    if (pKF == this)
    {
        spdlog::error("Cannot set a keyframe as its own parent");
        throw std::invalid_argument("The parent and child can not be the same");
    }

    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame *> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame *KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::SetFirstConnection(bool bFirst)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection = bFirst;
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame *> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::AddMergeEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<KeyFrame *> KeyFrame::GetMergeEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspMergeEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if (mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mnId == mpMap->get_initial_key_frame_id())
        {
            return;
        }
        else if (mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for (std::map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
    {
        mit->first->EraseConnection(this);
    }

    for (size_t i = 0; i < mvpMapPoints.size(); i++)
    {
        if (mvpMapPoints[i])
        {
            mvpMapPoints[i]->erase_observation(this);
        }
    }

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame *> sParentCandidates;
        if (mpParent)
        {
            sParentCandidates.insert(mpParent);
        }

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while (!mspChildrens.empty())
        {
            bool bContinue = false;

            int       max = -1;
            KeyFrame *pC  = nullptr;
            KeyFrame *pP  = nullptr;

            for (set<KeyFrame *>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
            {
                KeyFrame *pKF = *sit;
                if (pKF->isBad())
                {
                    continue;
                }

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame *> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
                {
                    for (set<KeyFrame *>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
                    {
                        if (vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if (w > max)
                            {
                                pC        = pKF;
                                pP        = vpConnected[i];
                                max       = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if (bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
            {
                break;
            }
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if (!mspChildrens.empty())
        {
            for (set<KeyFrame *>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }
        }

        if (mpParent)
        {
            mpParent->EraseChild(this);
            mTcp = mTcw * mpParent->GetPoseInverse();
        }
        mbBad = true;
    }

    mpMap->erase_key_frame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame *pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
    {
        UpdateBestCovisibles();
    }
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
    if (nMinCellX >= mnGridCols)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)mnGridCols - 1, (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
    if (nMinCellY >= mnGridRows)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)mnGridRows - 1, (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
    {
        return vIndices;
    }

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const auto &kpUn  = (!is_fisheye_stereo(cameras) || !bRight) ? mvKeys[vCell[j]] : mvKeysRight[vCell[j]];
                const float distx = kpUn.pt_undistorted.x - x;
                const float disty = kpUn.pt_undistorted.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                {
                    vIndices.push_back(vCell[j]);
                }
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const { return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY); }

bool KeyFrame::UnprojectStereo(int i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if (z > 0)
    {
        const float     u = mvKeys[i].pt.x;
        const float     v = mvKeys[i].pt.y;
        const float     x = (u - cx) * z * invfx;
        const float     y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        x3D = mRwc * x3Dc + mTwc.translation();
        return true;
    }
    else
    {
        return false;
    }
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    if (N == 0)
    {
        return -1.0;
    }

    vector<map_point *> vpMapPoints;
    Eigen::Matrix3f     Rcw;
    Eigen::Vector3f     tcw;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        tcw         = mTcw.translation();
        Rcw         = mRcw;
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    Eigen::Matrix<float, 1, 3> Rcw2 = Rcw.row(2);
    float                      zcw  = tcw(2);
    for (int i = 0; i < N; i++)
    {
        if (mvpMapPoints[i])
        {
            map_point      *pMP  = mvpMapPoints[i];
            Eigen::Vector3f x3Dw = pMP->get_world_position();
            float           z    = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(), vDepths.end());

    return vDepths[(vDepths.size() - 1) / q];
}

void KeyFrame::SetNewBias(const IMU::Bias &b)
{
    unique_lock<mutex> lock(mMutexPose);
    mImuBias = b;
    if (mpImuPreintegrated)
    {
        mpImuPreintegrated->SetNewBias(b);
    }
}

Eigen::Vector3f KeyFrame::GetGyroBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

Eigen::Vector3f KeyFrame::GetAccBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias KeyFrame::GetImuBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}

map *KeyFrame::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void KeyFrame::UpdateMap(map *pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

bool KeyFrame::ProjectPointDistort(map_point *pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->get_world_position();

    // 3D in camera coordinates
    Eigen::Vector3f Pc  = mRcw * P + mTcw.translation();
    float          &PcX = Pc(0);
    float          &PcY = Pc(1);
    float          &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
        spdlog::debug("Negative depth: {}", PcZ);
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f / PcZ;
    u          = fx * PcX * invz + cx;
    v          = fy * PcY * invz + cy;

    if (u < mnMinX || u > mnMaxX)
    {
        return false;
    }
    if (v < mnMinY || v > mnMaxY)
    {
        return false;
    }

    float x  = (u - cx) * invfx;
    float y  = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if (mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    float u_distort = x_distort * fx + cx;
    float v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool KeyFrame::ProjectPointUnDistort(map_point *pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->get_world_position();

    // 3D in camera coordinates
    Eigen::Vector3f Pc  = mRcw * P + mTcw.translation();
    float          &PcX = Pc(0);
    float          &PcY = Pc(1);
    float          &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
        spdlog::debug("Negative depth: {}", PcZ);
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    u                = fx * PcX * invz + cx;
    v                = fy * PcY * invz + cy;

    if (u < mnMinX || u > mnMaxX)
    {
        return false;
    }
    if (v < mnMinY || v > mnMaxY)
    {
        return false;
    }

    kp = cv::Point2f(u, v);

    return true;
}

Sophus::SE3f KeyFrame::GetRelativePoseTrl()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTrl;
}

Sophus::SE3f KeyFrame::GetRelativePoseTlr()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTlr;
}

Sophus::SE3<float> KeyFrame::GetRightPose()
{
    unique_lock<mutex> lock(mMutexPose);

    return mTrl * mTcw;
}

Sophus::SE3<float> KeyFrame::GetRightPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);

    return mTwc * mTlr;
}

Eigen::Vector3f KeyFrame::GetRightCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);

    return (mTwc * mTlr).translation();
}

Eigen::Matrix<float, 3, 3> KeyFrame::GetRightRotation()
{
    unique_lock<mutex> lock(mMutexPose);

    return (mTrl.so3() * mTcw.so3()).matrix();
}

Eigen::Vector3f KeyFrame::GetRightTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mTrl * mTcw).translation();
}

void KeyFrame::SetKeyFrameDatabase(key_frame_database *pKFDB) { mpKeyFrameDB = pKFDB; }

} // namespace ORB_SLAM3
