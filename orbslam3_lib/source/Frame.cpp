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

#include "Frame.h"

#include "CameraModels/KannalaBrandt8.h"
#include "CameraModels/Pinhole.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "descriptor_matcher.h"
#include "keypoint_extractor.h"

#include <array>
#include <thread>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId               = 0;
bool              Frame::mbInitialComputations = true;
float             Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float             Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float             Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

// For stereo fisheye matching

Frame::Frame()
    : mpcpi(nullptr), mbHasPose(false), mbHasVelocity(false), mpImuPreintegrated(nullptr), mpPrevFrame(nullptr), mpImuPreintegratedFrame(nullptr), mpReferenceKF(nullptr), mbIsSet(false),
      mbImuPreintegrated(false)
{
}

// Copy Constructor
Frame::Frame(const Frame &frame)
    : mpcpi(frame.mpcpi), mTcw(frame.mTcw), mbHasPose(false), mTlr(frame.mTlr), mTrl(frame.mTrl), mRlr(frame.mRlr), mtlr(frame.mtlr), mbHasVelocity(false),
      place_vocabulary_ptr(frame.place_vocabulary_ptr), left_keypoint_extractor(frame.left_keypoint_extractor), right_keypoint_extractor(frame.right_keypoint_extractor),
      mTimeStamp(frame.mTimeStamp), optical_flow_pyramid(frame.optical_flow_pyramid), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
      mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys), mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), display_keys(frame.display_keys),
      display_keys_right(frame.display_keys_right), mvpMapPoints(frame.mvpMapPoints), mvuRight(frame.mvuRight), mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
      mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()), mvbOutlier(frame.mvbOutlier), mnCloseMPs(frame.mnCloseMPs), mImuBias(frame.mImuBias),
      mImuCalib(frame.mImuCalib), mpImuPreintegrated(frame.mpImuPreintegrated), mpLastKeyFrame(frame.mpLastKeyFrame), mpPrevFrame(frame.mpPrevFrame),
      mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
      mfLogScaleFactor(frame.mfLogScaleFactor), mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2),
      mvInvLevelSigma2(frame.mvInvLevelSigma2), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset), mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated),
      mpMutexImu(frame.mpMutexImu), Nleft(frame.Nleft), Nright(frame.Nright), monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
      mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints)
{
    flow_only               = frame.flow_only;
    flow_candidates         = frame.flow_candidates;
    flow_tracks             = frame.flow_tracks;
    flow_inliers            = frame.flow_inliers;
    flow_occupied_cells     = frame.flow_occupied_cells;
    place_descriptors       = frame.place_descriptors.clone();
    place_descriptors_right = frame.place_descriptors_right.clone();
    for (int i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
        {
            mGrid[i][j] = frame.mGrid[i][j];
            if (frame.Nleft > 0)
            {
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }
    }

    if (frame.mbHasPose)
    {
        SetPose(frame.GetPose());
    }

    if (frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints  = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;
}

const std::vector<cv::Mat> &Frame::get_optical_flow_pyramid(const cv::Mat &image)
{
    if (optical_flow_pyramid.empty())
    {
        optical_flow_pyramid = build_optical_flow_pyramid(image);
    }

    return optical_flow_pyramid;
}

std::vector<cv::Mat> Frame::build_optical_flow_pyramid(const cv::Mat &image)
{
    std::vector<cv::Mat> pyramid;
    cv::buildOpticalFlowPyramid(image, pyramid, cv::Size(21, 21), 3, true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
    return pyramid;
}

Frame::Frame(Frame &previous, double timestamp) : Frame(static_cast<const Frame &>(previous))
{
    mnId                    = nNextId++;
    mTimeStamp              = timestamp;
    mpPrevFrame             = &previous;
    mpcpi                   = nullptr;
    mpImuPreintegrated      = nullptr;
    mpImuPreintegratedFrame = nullptr;
    mbImuPreintegrated      = false;
    mbHasPose               = false;
    mbIsSet                 = false;
    flow_candidates         = 0;
    flow_tracks             = 0;
    flow_inliers            = 0;
    flow_occupied_cells     = 0;

    optical_flow_pyramid.clear();
    imgLeft.release();
    imgRight.release();

    set_tracked_features({}, {});
}

void Frame::set_tracked_features(std::vector<cv::KeyPoint> keys, std::vector<MapPoint *> points, std::vector<cv::KeyPoint> native_keys)
{
    CV_Assert(keys.size() == points.size());
    CV_Assert(native_keys.empty() || native_keys.size() == keys.size());
    flow_only    = true;
    mvKeys       = std::move(keys);
    display_keys = native_keys.empty() ? mvKeys : std::move(native_keys);
    display_keys_right.clear();
    mvpMapPoints = std::move(points);
    N            = static_cast<int>(mvKeys.size());
    Nleft        = Nleft != -1 ? N : -1;
    Nright       = Nleft != -1 ? 0 : -1;
    monoLeft = monoRight = -1;
    mvKeysRight.clear();
    mDescriptors.release();
    mDescriptorsRight.release();
    place_descriptors.release();
    place_descriptors_right.release();
    mBowVec.clear();
    mFeatVec.clear();
    mvbOutlier.assign(N, false);
    mvuRight.assign(N, -1.0f);
    mvDepth.assign(N, -1.0f);
    mnCloseMPs = 0;
    mvLeftToRightMatch.assign(Nleft != -1 ? N : 0, -1);
    mvRightToLeftMatch.clear();
    mvStereo3Dpoints.clear();
    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    // Fisheye observations remain in native camera coordinates.
    if (Nleft != -1 || N == 0)
    {
        mvKeysUn = mvKeys;
    }
    else
    {
        UndistortKeyPoints();
    }
    for (int x = 0; x < FRAME_GRID_COLS; ++x)
    {
        for (int y = 0; y < FRAME_GRID_ROWS; ++y)
        {
            mGrid[x][y].clear();
            mGridRight[x][y].clear();
        }
    }
    AssignFeaturesToGrid();
}

// TODO: why is the `Frame` class extracting features? Seems like it should be the responsibility of a separate class or module dedicated to feature extraction.
void Frame::extract_features(const cv::Mat &left, GeometricCamera *camera, const cv::Mat &right, GeometricCamera *right_camera)
{
    flow_only = false;
    mvKeys.clear();
    mvKeysRight.clear();
    display_keys.clear();
    display_keys_right.clear();
    mDescriptors.release();
    mDescriptorsRight.release();
    place_descriptors.release();
    place_descriptors_right.release();
    mBowVec.clear();
    mFeatVec.clear();
    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    for (int x = 0; x < FRAME_GRID_COLS; ++x)
    {
        for (int y = 0; y < FRAME_GRID_ROWS; ++y)
        {
            mGrid[x][y].clear();
            mGridRight[x][y].clear();
        }
    }
    if (right.empty())
    {
        extract_keypoints(0, left, 0, 1000);
    }
    else
    {
        const auto *left_camera          = right_camera ? static_cast<KannalaBrandt8 *>(camera) : nullptr;
        const auto *right_fisheye_camera = right_camera ? static_cast<KannalaBrandt8 *>(right_camera) : nullptr;
        thread      left_worker(&Frame::extract_keypoints, this, 0, left, left_camera ? left_camera->mvLappingArea[0] : 0, left_camera ? left_camera->mvLappingArea[1] : 0);
        thread      right_worker(&Frame::extract_keypoints, this, 1, right, right_fisheye_camera ? right_fisheye_camera->mvLappingArea[0] : 0,
                            right_fisheye_camera ? right_fisheye_camera->mvLappingArea[1] : 0);
        left_worker.join();
        right_worker.join();
    }
    Nleft  = right_camera ? static_cast<int>(mvKeys.size()) : -1;
    Nright = right_camera ? static_cast<int>(mvKeysRight.size()) : -1;
    N      = static_cast<int>(mvKeys.size() + (right_camera ? mvKeysRight.size() : 0));
    mvpMapPoints.assign(N, nullptr);
    mvbOutlier.assign(N, false);
    mvuRight.assign(mvKeys.size(), -1.0f);
    mvDepth.assign(mvKeys.size(), -1.0f);
    mvLeftToRightMatch.clear();
    mvRightToLeftMatch.clear();
    mvStereo3Dpoints.clear();
    mnCloseMPs = 0;
    if (!right_camera)
    {
        monoLeft = monoRight = -1;
    }
    if (N == 0)
    {
        mvKeysUn.clear();
        return;
    }
    if (mbInitialComputations)
    {
        ComputeImageBounds(left);
        mfGridElementWidthInv  = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);
        fx                     = mK.at<float>(0, 0);
        fy                     = mK.at<float>(1, 1);
        cx                     = mK.at<float>(0, 2);
        cy                     = mK.at<float>(1, 2);
        invfx                  = 1.0f / fx;
        invfy                  = 1.0f / fy;
        mbInitialComputations  = false;
    }
    mb = mbf / fx;
    if (right_camera)
    {
        imgLeft  = left.clone();
        imgRight = right.clone();
        ComputeStereoFishEyeMatches(camera, right_camera);
        cv::vconcat(mDescriptors, mDescriptorsRight, mDescriptors);
        cv::vconcat(place_descriptors, place_descriptors_right, place_descriptors);
    }
    else if (!right.empty())
    {
        ComputeStereoMatches();
    }
    if (right_camera)
    {
        mvKeysUn = mvKeys;
    }
    else
    {
        UndistortKeyPoints();
    }
    AssignFeaturesToGrid();
}

bool Frame::reliable_flow_inliers(const cv::Size &image_size, int minimum_inliers)
{
    flow_inliers        = 0;
    flow_occupied_cells = 0;

    if (image_size.width <= 0 || image_size.height <= 0 || minimum_inliers < 50)
    {
        return false;
    }

    std::array<bool, 16> occupied{};
    int                  inliers = 0;

    for (std::size_t index = 0; index < mvKeys.size(); ++index)
    {
        if (!mvpMapPoints[index] || mvbOutlier[index])
        {
            continue;
        }

        const auto &point = mvKeys[index].pt;

        if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 || point.y < 0 || point.x >= image_size.width || point.y >= image_size.height)
        {
            continue;
        }

        const int x = static_cast<int>(point.x * 4 / image_size.width);
        const int y = static_cast<int>(point.y * 4 / image_size.height);

        occupied[static_cast<std::size_t>(y * 4 + x)] = true;

        ++inliers;
    }
    flow_inliers        = inliers;
    flow_occupied_cells = static_cast<int>(std::count(occupied.begin(), occupied.end(), true));

    return inliers >= minimum_inliers && inliers >= 0.6 * N && flow_occupied_cells >= 6;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, keypoint_extractor *extractorLeft, keypoint_extractor *extractorRight, place_vocabulary *voc, cv::Mat &K,
             cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(nullptr), mbHasPose(false), mbHasVelocity(false), place_vocabulary_ptr(voc), left_keypoint_extractor(extractorLeft), right_keypoint_extractor(extractorRight),
      mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL),
      mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false)
{
    // Frame ID
    mnId              = nNextId++;

    // Scale Level Info
    mnScaleLevels     = left_keypoint_extractor->get_levels();
    mfScaleFactor     = left_keypoint_extractor->get_scale_factor();
    mfLogScaleFactor  = log(mfScaleFactor);
    mvScaleFactors    = left_keypoint_extractor->get_scale_factors();
    mvInvScaleFactors = left_keypoint_extractor->get_inverse_scale_factors();
    mvLevelSigma2     = left_keypoint_extractor->get_scale_sigma_squares();
    mvInvLevelSigma2  = left_keypoint_extractor->get_inverse_scale_sigma_squares();

    extract_features(imLeft, pCamera, imRight);

    if (pPrevF)
    {
        if (pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, keypoint_extractor *extractor, place_vocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
             const float &thDepth, Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(nullptr), mbHasPose(false), mbHasVelocity(false), place_vocabulary_ptr(voc), left_keypoint_extractor(extractor), right_keypoint_extractor(static_cast<keypoint_extractor *>(NULL)),
      mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL),
      mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false)
{
    // Frame ID
    mnId              = nNextId++;

    // Scale Level Info
    mnScaleLevels     = left_keypoint_extractor->get_levels();
    mfScaleFactor     = left_keypoint_extractor->get_scale_factor();
    mfLogScaleFactor  = log(mfScaleFactor);
    mvScaleFactors    = left_keypoint_extractor->get_scale_factors();
    mvInvScaleFactors = left_keypoint_extractor->get_inverse_scale_factors();
    mvLevelSigma2     = left_keypoint_extractor->get_scale_sigma_squares();
    mvInvLevelSigma2  = left_keypoint_extractor->get_inverse_scale_sigma_squares();

    // ORB extraction
    extract_keypoints(0, imGray, 0, 0);

    N = mvKeys.size();

    if (mvKeys.empty())
    {
        return;
    }

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv  = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx                     = K.at<float>(0, 0);
        fy                     = K.at<float>(1, 1);
        cx                     = K.at<float>(0, 2);
        cy                     = K.at<float>(1, 2);
        invfx                  = 1.0f / fx;
        invfy                  = 1.0f / fy;

        mbInitialComputations  = false;
    }

    mb = mbf / fx;

    if (pPrevF)
    {
        if (pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    // Set no stereo fisheye information
    Nleft              = -1;
    Nright             = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints   = vector<Eigen::Vector3f>(0);
    monoLeft           = -1;
    monoRight          = -1;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const double &timeStamp, keypoint_extractor *extractor, place_vocabulary *voc, GeometricCamera *pCamera, cv::Mat &distCoef, const float &bf,
             const float &thDepth, Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(nullptr), mbHasPose(false), mbHasVelocity(false), place_vocabulary_ptr(voc), left_keypoint_extractor(extractor), right_keypoint_extractor(static_cast<keypoint_extractor *>(NULL)),
      mTimeStamp(timeStamp), mK(static_cast<Pinhole *>(pCamera)->toK()), mK_(static_cast<Pinhole *>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
      mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false)
{
    // Frame ID
    mnId              = nNextId++;

    // Scale Level Info
    mnScaleLevels     = left_keypoint_extractor->get_levels();
    mfScaleFactor     = left_keypoint_extractor->get_scale_factor();
    mfLogScaleFactor  = log(mfScaleFactor);
    mvScaleFactors    = left_keypoint_extractor->get_scale_factors();
    mvInvScaleFactors = left_keypoint_extractor->get_inverse_scale_factors();
    mvLevelSigma2     = left_keypoint_extractor->get_scale_sigma_squares();
    mvInvLevelSigma2  = left_keypoint_extractor->get_inverse_scale_sigma_squares();

    extract_features(imGray, pCamera);

    if (pPrevF)
    {
        if (pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }
}

void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells   = FRAME_GRID_COLS * FRAME_GRID_ROWS;

    int       nReserve = 0.5f * N / (nCells);

    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
        {
            mGrid[i][j].reserve(nReserve);
            if (Nleft != -1)
            {
                mGridRight[i][j].reserve(nReserve);
            }
        }
    }

    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i] : (i < Nleft) ? mvKeys[i] : mvKeysRight[i - Nleft];

        int                 nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
        {
            if (Nleft == -1 || i < Nleft)
            {
                mGrid[nGridPosX][nGridPosY].push_back(i);
            }
            else
            {
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
            }
        }
    }
}

void Frame::extract_keypoints(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0, x1};
    if (flag == 0)
    {
        monoLeft          = (*left_keypoint_extractor)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);
        display_keys      = left_keypoint_extractor->get_display_keypoints();
        place_descriptors = left_keypoint_extractor->get_place_descriptors().clone();
    }
    else
    {
        monoRight               = (*right_keypoint_extractor)(im, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);
        display_keys_right      = right_keypoint_extractor->get_display_keypoints();
        place_descriptors_right = right_keypoint_extractor->get_place_descriptors().clone();
    }
}

bool Frame::isSet() const { return mbIsSet; }

void Frame::SetPose(const Sophus::SE3<float> &Tcw)
{
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet   = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if (mpImuPreintegrated)
    {
        mpImuPreintegrated->SetNewBias(b);
    }
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw           = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const { return mVw; }

void            Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw           = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw             = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet   = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc                   = Twc.rotationMatrix();
    mOw                    = Twc.translation();
    mRcw                   = mTcw.rotationMatrix();
    mtcw                   = mTcw.translation();
}

Eigen::Matrix<float, 3, 1> Frame::GetImuPosition() const { return mRwc * mImuCalib.mTcb.translation() + mOw; }

Eigen::Matrix<float, 3, 3> Frame::GetImuRotation() { return mRwc * mImuCalib.mTcb.rotationMatrix(); }

Sophus::SE3<float>         Frame::GetImuPose() { return mTcw.inverse() * mImuCalib.mTcb; }

Sophus::SE3f               Frame::GetRelativePoseTrl() { return mTrl; }

Sophus::SE3f               Frame::GetRelativePoseTlr() { return mTlr; }

Eigen::Matrix3f            Frame::GetRelativePoseTlr_rotation() { return mTlr.rotationMatrix(); }

Eigen::Vector3f            Frame::GetRelativePoseTlr_translation() { return mTlr.translation(); }

bool                       Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit, GeometricCamera *camera, GeometricCamera *right_camera)
{
    if (Nleft == -1)
    {
        pMP->mbTrackInView                       = false;
        pMP->mTrackProjX                         = -1;
        pMP->mTrackProjY                         = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float, 3, 1>       P       = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float, 3, 1> Pc      = mRcw * P + mtcw;
        const float                      Pc_dist = Pc.norm();

        // Check positive depth
        const float                     &PcZ     = Pc(2);
        const float                      invz    = 1.0f / PcZ;
        if (PcZ < 0.0f)
        {
            return false;
        }

        const Eigen::Vector2f uv = camera->project(Pc);

        if (uv(0) < mnMinX || uv(0) > mnMaxX)
        {
            return false;
        }
        if (uv(1) < mnMinY || uv(1) > mnMaxY)
        {
            return false;
        }

        pMP->mTrackProjX                  = uv(0);
        pMP->mTrackProjY                  = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float           maxDistance = pMP->GetMaxDistanceInvariance();
        const float           minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO          = P - mOw;
        const float           dist        = PO.norm();

        if (dist < minDistance || dist > maxDistance)
        {
            return false;
        }

        // Check viewing angle
        Eigen::Vector3f Pn      = pMP->GetNormal();

        const float     viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
        {
            return false;
        }

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView        = true;
        pMP->mTrackProjX          = uv(0);
        pMP->mTrackProjXR         = uv(0) - mbf * invz;

        pMP->mTrackDepth          = Pc_dist;

        pMP->mTrackProjY          = uv(1);
        pMP->mnTrackScaleLevel    = nPredictedLevel;
        pMP->mTrackViewCos        = viewCos;

        return true;
    }
    else
    {
        pMP->mbTrackInView      = false;
        pMP->mbTrackInViewR     = false;
        pMP->mnTrackScaleLevel  = -1;
        pMP->mnTrackScaleLevelR = -1;

        pMP->mbTrackInView      = isInFrustumChecks(pMP, viewingCosLimit, camera);
        pMP->mbTrackInViewR     = isInFrustumChecks(pMP, viewingCosLimit, right_camera, true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint *pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f       P   = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc  = mRcw * P + mtcw;
    const float          &PcX = Pc(0);
    const float          &PcY = Pc(1);
    const float          &PcZ = Pc(2);

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

    float u_distort, v_distort;

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
    x_distort       = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort       = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort       = x_distort * fx + cx;
    v_distort       = y_distort * fy + cy;

    u               = u_distort;
    v               = v_distort;

    kp              = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw) { return mRcw * pCw + mtcw; }

vector<size_t>  Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float     factorX   = r;
    float     factorY   = r;

    const int nMinCellX = max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if (vCell.empty())
            {
                continue;
            }

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]] : (!bRight) ? mvKeys[vCell[j]] : mvKeysRight[vCell[j]];
                if (bCheckLevels)
                {
                    if (kpUn.octave < minLevel)
                    {
                        continue;
                    }
                    if (maxLevel >= 0)
                    {
                        if (kpUn.octave > maxLevel)
                        {
                            continue;
                        }
                    }
                }

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx) < factorX && fabs(disty) < factorY)
                {
                    vIndices.push_back(vCell[j]);
                }
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
    {
        return false;
    }

    return true;
}

void Frame::ComputeBoW()
{
    if (mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(place_descriptors);
        place_vocabulary_ptr->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::UndistortKeyPoints()
{
    if (mDistCoef.at<float>(0) == 0.0)
    {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);

    for (int i = 0; i < N; i++)
    {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x         = mat.at<float>(i, 0);
        kp.pt.y         = mat.at<float>(i, 1);
        mvKeysUn[i]     = kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if (mDistCoef.at<float>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols;
        mat.at<float>(3, 1) = imLeft.rows;

        mat                 = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat    = mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight                                    = vector<float>(N, -1.0f);
    mvDepth                                     = vector<float>(N, -1.0f);

    const int              descriptor_threshold = (descriptor_matcher::high_distance_threshold + descriptor_matcher::low_distance_threshold) / 2;

    const int              nRows                = left_keypoint_extractor->mvImagePyramid[0].rows;

    // Assign keypoints to row table
    vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());

    for (int i = 0; i < nRows; i++)
    {
        vRowIndices[i].reserve(200);
    }

    const int Nr = mvKeysRight.size();

    for (int iR = 0; iR < Nr; iR++)
    {
        const cv::KeyPoint &kp   = mvKeysRight[iR];
        const float        &kpY  = kp.pt.y;
        const float         r    = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
        const int           maxr = ceil(kpY + r);
        const int           minr = floor(kpY - r);

        for (int yi = minr; yi <= maxr; yi++)
        {
            vRowIndices[yi].push_back(iR);
        }
    }

    // Set limits for search
    const float            minZ = mb;
    const float            minD = 0;
    const float            maxD = mbf / minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int>> vDistIdx;
    vDistIdx.reserve(N);

    for (int iL = 0; iL < N; iL++)
    {
        const cv::KeyPoint   &kpL         = mvKeys[iL];
        const int            &levelL      = kpL.octave;
        const float          &vL          = kpL.pt.y;
        const float          &uL          = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if (vCandidates.empty())
        {
            continue;
        }

        const float minU = uL - maxD;
        const float maxU = uL - minD;

        if (maxU < 0)
        {
            continue;
        }

        float          bestDist          = descriptor_matcher::high_distance_threshold;
        size_t         bestIdxR          = 0;

        const cv::Mat &dL                = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        const auto     ranked_candidates = descriptor_matcher::rank_candidates(dL, mDescriptorsRight, vCandidates);
        for (const auto iR : ranked_candidates)
        {
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
            {
                continue;
            }

            const float &uR = kpR.pt.x;

            if (uR >= minU && uR <= maxU)
            {
                const cv::Mat &dR   = mDescriptorsRight.row(iR);
                const float    dist = descriptor_matcher::descriptor_distance(dL, dR);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if (bestDist < descriptor_threshold)
        {
            // coordinates in image pyramid at keypoint scale
            const float   uR0         = mvKeysRight[bestIdxR].pt.x;
            const float   scaleFactor = mvInvScaleFactors[kpL.octave];
            const float   scaleduL    = round(kpL.pt.x * scaleFactor);
            const float   scaledvL    = round(kpL.pt.y * scaleFactor);
            const float   scaleduR0   = round(uR0 * scaleFactor);

            // sliding window search
            const int     w           = 5;
            cv::Mat       IL          = left_keypoint_extractor->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);

            float         bestDist    = INT_MAX;
            int           bestincR    = 0;
            const int     L           = 5;
            vector<float> vDists;
            vDists.resize(2 * L + 1);

            const float iniu = scaleduR0 + L - w;
            const float endu = scaleduR0 + L + w + 1;
            if (iniu < 0 || endu >= right_keypoint_extractor->mvImagePyramid[kpL.octave].cols)
            {
                continue;
            }

            for (int incR = -L; incR <= +L; incR++)
            {
                cv::Mat IR   = right_keypoint_extractor->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

                float   dist = cv::norm(IL, IR, cv::NORM_L1);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestincR = incR;
                }

                vDists[L + incR] = dist;
            }

            if (bestincR == -L || bestincR == L)
            {
                continue;
            }

            // Sub-pixel match (Parabola fitting)
            const float dist1  = vDists[L + bestincR - 1];
            const float dist2  = vDists[L + bestincR];
            const float dist3  = vDists[L + bestincR + 1];

            const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

            if (deltaR < -1 || deltaR > 1)
            {
                continue;
            }

            // Re-scaled coordinate
            float bestuR    = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

            float disparity = (uL - bestuR);

            if (disparity >= minD && disparity < maxD)
            {
                if (disparity <= 0)
                {
                    disparity = 0.01;
                    bestuR    = uL - 0.01;
                }
                mvDepth[iL]  = mbf / disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int, int>(bestDist, iL));
            }
        }
    }

    sort(vDistIdx.begin(), vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;

    for (int i = vDistIdx.size() - 1; i >= 0; i--)
    {
        if (vDistIdx[i].first < thDist)
        {
            break;
        }
        else
        {
            mvuRight[vDistIdx[i].second] = -1;
            mvDepth[vDistIdx[i].second]  = -1;
        }
    }
}

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N, -1);
    mvDepth  = vector<float>(N, -1);

    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp  = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float        &v   = kp.pt.y;
        const float        &u   = kp.pt.x;

        const float         d   = imDepth.at<float>(v, u);

        if (d > 0)
        {
            mvDepth[i]  = d;
            mvuRight[i] = kpU.pt.x - mbf / d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if (z > 0)
    {
        const float     u = mvKeysUn[i].pt.x;
        const float     v = mvKeysUn[i].pt.y;
        const float     x = (u - cx) * z * invfx;
        const float     y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    }
    else
    {
        return false;
    }
}

bool Frame::imuIsPreintegrated() const
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, keypoint_extractor *extractorLeft, keypoint_extractor *extractorRight, place_vocabulary *voc, cv::Mat &K,
             cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera *pCamera, GeometricCamera *pCamera2, Sophus::SE3f &Tlr, Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(nullptr), mbHasPose(false), mbHasVelocity(false), place_vocabulary_ptr(voc), left_keypoint_extractor(extractorLeft), right_keypoint_extractor(extractorRight),
      mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL),
      mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false)

{
    // Frame ID
    mnId              = nNextId++;

    // Scale Level Info
    mnScaleLevels     = left_keypoint_extractor->get_levels();
    mfScaleFactor     = left_keypoint_extractor->get_scale_factor();
    mfLogScaleFactor  = log(mfScaleFactor);
    mvScaleFactors    = left_keypoint_extractor->get_scale_factors();
    mvInvScaleFactors = left_keypoint_extractor->get_inverse_scale_factors();
    mvLevelSigma2     = left_keypoint_extractor->get_scale_sigma_squares();
    mvInvLevelSigma2  = left_keypoint_extractor->get_inverse_scale_sigma_squares();

    // Sophus/Eigen
    mTlr              = Tlr;
    mTrl              = mTlr.inverse();
    mRlr              = mTlr.rotationMatrix();
    mtlr              = mTlr.translation();

    extract_features(imLeft, pCamera, imRight, pCamera2);
}

void Frame::ComputeStereoFishEyeMatches(GeometricCamera *camera, GeometricCamera *right_camera)
{
    // Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat              stereoDescLeft  = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat              stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch                   = vector<int>(Nleft, -1);
    mvRightToLeftMatch                   = vector<int>(Nright, -1);
    mvDepth                              = vector<float>(Nleft, -1.0f);
    mvuRight                             = vector<float>(Nleft, -1);
    mvStereo3Dpoints                     = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs                           = 0;

    // Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    descriptor_matcher::knn_match(stereoDescLeft, stereoDescRight, matches, 2);

    int nMatches    = 0;
    int descMatches = 0;

    // Check matches using Lowe's ratio
    for (vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it)
    {
        if ((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7)
        {
            // For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8 *>(camera)->TriangulateMatches(right_camera, mvKeys[(*it)[0].queryIdx + monoLeft], mvKeysRight[(*it)[0].trainIdx + monoRight], mRlr, mtlr,
                                                                                    sigma1, sigma2, p3D);
            if (depth > 0.0001f)
            {
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft]  = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft]    = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft]             = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, GeometricCamera *camera, bool bRight)
{
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if (bRight)
    {
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR                  = Rrl * mRcw;
        mt                  = Rrl * mtcw + trl;
        twc                 = mRwc * mTlr.translation() + mOw;
    }
    else
    {
        mR  = mRcw;
        mt  = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc      = mR * P + mt;
    const float     Pc_dist = Pc.norm();
    const float    &PcZ     = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
        return false;
    }

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    uv = camera->project(Pc);

    if (uv(0) < mnMinX || uv(0) > mnMaxX)
    {
        return false;
    }
    if (uv(1) < mnMinY || uv(1) > mnMaxY)
    {
        return false;
    }

    // Check distance is in the scale invariance region of the MapPoint
    const float           maxDistance = pMP->GetMaxDistanceInvariance();
    const float           minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO          = P - twc;
    const float           dist        = PO.norm();

    if (dist < minDistance || dist > maxDistance)
    {
        return false;
    }

    // Check viewing angle
    Eigen::Vector3f Pn      = pMP->GetNormal();

    const float     viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit)
    {
        return false;
    }

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist, this);

    if (bRight)
    {
        pMP->mTrackProjXR       = uv(0);
        pMP->mTrackProjYR       = uv(1);
        pMP->mnTrackScaleLevelR = nPredictedLevel;
        pMP->mTrackViewCosR     = viewCos;
        pMP->mTrackDepthR       = Pc_dist;
    }
    else
    {
        pMP->mTrackProjX       = uv(0);
        pMP->mTrackProjY       = uv(1);
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos     = viewCos;
        pMP->mTrackDepth       = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i) { return mRwc * mvStereo3Dpoints[i] + mOw; }

} // namespace ORB_SLAM3
