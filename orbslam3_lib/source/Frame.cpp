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
#include "Converter.h"
#include "G2oTypes.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "descriptor_matcher.h"

#include <array>
#include <chrono>
#include <cmath>

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
      place_vocabulary_ptr(frame.place_vocabulary_ptr), mTimeStamp(frame.mTimeStamp), optical_flow_pyramid(frame.optical_flow_pyramid), mK(frame.mK.clone()),
      mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()), mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
      mvKeysRight(frame.mvKeysRight), is_fisheye_stereo(frame.is_fisheye_stereo), mvpMapPoints(frame.mvpMapPoints), mvuRight(frame.mvuRight), mvDepth(frame.mvDepth), mBowVec(frame.mBowVec),
      mFeatVec(frame.mFeatVec), mvbOutlier(frame.mvbOutlier), mnCloseMPs(frame.mnCloseMPs), mImuBias(frame.mImuBias), mImuCalib(frame.mImuCalib), mpImuPreintegrated(frame.mpImuPreintegrated),
      mpLastKeyFrame(frame.mpLastKeyFrame), mpPrevFrame(frame.mpPrevFrame), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF),
      mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor), mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
      mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu), monoLeft(frame.monoLeft), monoRight(frame.monoRight),
      mvLeftToRightMatch(frame.mvLeftToRightMatch), mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints)
{
    color_image         = frame.color_image;
    color_image_right   = frame.color_image_right;
    color_is_rgb        = frame.color_is_rgb;
    flow_only           = frame.flow_only;
    flow_candidates     = frame.flow_candidates;
    flow_tracks         = frame.flow_tracks;
    flow_inliers        = frame.flow_inliers;
    flow_occupied_cells = frame.flow_occupied_cells;
    extraction_ms       = frame.extraction_ms;
    stereo_matching_ms  = frame.stereo_matching_ms;
    optical_flow_ms     = frame.optical_flow_ms;
    for (int i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
        {
            mGrid[i][j] = frame.mGrid[i][j];
            if (frame.is_fisheye_stereo)
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

std::array<unsigned char, 3> Frame::keypoint_color(int index) const
{
    const bool  right     = is_fisheye_stereo && index >= static_cast<int>(mvKeys.size());
    const auto &image     = right ? color_image_right : color_image;
    const auto &keys      = right ? mvKeysRight : mvKeys;
    const int   key_index = right ? index - static_cast<int>(mvKeys.size()) : index;
    if (image.empty() || image.depth() != CV_8U || key_index < 0 || static_cast<std::size_t>(key_index) >= keys.size())
    {
        return {255, 255, 255};
    }
    const auto &position = keys[key_index].pt;
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || position.x < 0 || position.y < 0 || position.x >= image.cols || position.y >= image.rows)
    {
        return {255, 255, 255};
    }
    const int   column = std::min(cvRound(position.x), image.cols - 1);
    const int   row    = std::min(cvRound(position.y), image.rows - 1);
    const auto *pixel  = image.ptr<unsigned char>(row) + column * image.channels();
    if (image.channels() == 1)
    {
        return {pixel[0], pixel[0], pixel[0]};
    }
    if (image.channels() == 3 || image.channels() == 4)
    {
        return color_is_rgb ? std::array<unsigned char, 3>{pixel[0], pixel[1], pixel[2]} : std::array<unsigned char, 3>{pixel[2], pixel[1], pixel[0]};
    }
    return {255, 255, 255};
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
    extraction_ms           = 0;
    stereo_matching_ms      = 0;
    optical_flow_ms         = 0;

    optical_flow_pyramid.clear();
    imgLeft.release();
    imgRight.release();

    set_tracked_features({}, {});
}

void Frame::set_tracked_features(std::vector<keypoint> keys, std::vector<MapPoint *> points)
{
    CV_Assert(keys.size() == points.size());
    flow_only = true;
    mvKeys    = std::move(keys);
    for (auto &key : mvKeys)
    {
        key.descriptor.release();
        key.place_descriptor.release();
    }
    mvpMapPoints = std::move(points);
    N            = static_cast<int>(mvKeys.size());
    monoLeft = monoRight = -1;
    mvKeysRight.clear();
    mBowVec.clear();
    mFeatVec.clear();
    mvbOutlier.assign(N, false);
    mvuRight.assign(N, -1.0f);
    mvDepth.assign(N, -1.0f);
    mnCloseMPs = 0;
    mvLeftToRightMatch.assign(is_fisheye_stereo ? N : 0, -1);
    mvRightToLeftMatch.clear();
    mvStereo3Dpoints.clear();
    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    // Fisheye observations remain in native camera coordinates.
    if (is_fisheye_stereo || N == 0)
    {
        for (auto &key : mvKeys)
        {
            key.pt_undistorted = key.pt;
        }
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

void Frame::initialize_features(const cv::Mat &left, const camera_vector &cameras, const cv::Mat &right, const std::vector<cv::Mat> &left_pyramid, const std::vector<cv::Mat> &right_pyramid)
{
    flow_only = false;
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
    is_fisheye_stereo = ORB_SLAM3::is_fisheye_stereo(cameras);
    N                 = static_cast<int>(mvKeys.size() + (is_fisheye_stereo ? mvKeysRight.size() : 0));
    mvpMapPoints.assign(N, nullptr);
    mvbOutlier.assign(N, false);
    mvuRight.assign(mvKeys.size(), -1.0f);
    mvDepth.assign(mvKeys.size(), -1.0f);
    mvLeftToRightMatch.clear();
    mvRightToLeftMatch.clear();
    mvStereo3Dpoints.clear();
    mnCloseMPs = 0;
    if (!is_fisheye_stereo)
    {
        monoLeft = monoRight = -1;
    }
    if (N == 0)
    {
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
    mb                        = mbf / fx;
    const auto stereo_started = std::chrono::steady_clock::now();
    if (is_fisheye_stereo)
    {
        imgLeft  = left.clone();
        imgRight = right.clone();
        ComputeStereoFishEyeMatches(cameras);
    }
    else if (!right.empty())
    {
        ComputeStereoMatches(left_pyramid, right_pyramid);
    }
    stereo_matching_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stereo_started).count();
    if (is_fisheye_stereo)
    {
        for (auto &key : mvKeys)
        {
            key.pt_undistorted = key.pt;
        }
        for (auto &key : mvKeysRight)
        {
            key.pt_undistorted = key.pt;
        }
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

Frame::Frame(double              timestamp,
             place_vocabulary   *vocabulary,
             const cv::Mat      &calibration,
             const cv::Mat      &distortion,
             float               baseline_fx,
             float               depth_threshold,
             Frame              *previous,
             const IMU::Calib   &imu_calibration,
             const Sophus::SE3f &left_to_right)
    : mpcpi(nullptr), mbHasPose(false), mTlr(left_to_right), mTrl(left_to_right.inverse()), mRlr(left_to_right.rotationMatrix()), mtlr(left_to_right.translation()), mbHasVelocity(false),
      place_vocabulary_ptr(vocabulary), mTimeStamp(timestamp), mK(calibration.clone()), mK_(Converter::toMatrix3f(calibration)), mDistCoef(distortion.clone()), mbf(baseline_fx), mb(0),
      mThDepth(depth_threshold), N(0), mnCloseMPs(0), mImuCalib(imu_calibration), mpImuPreintegrated(nullptr), mpLastKeyFrame(nullptr), mpPrevFrame(previous), mpImuPreintegratedFrame(nullptr),
      mnId(nNextId++), mpReferenceKF(nullptr), mnScaleLevels(0), mfScaleFactor(1), mfLogScaleFactor(0), mnDataset(0), mbIsSet(false), mbImuPreintegrated(false), monoLeft(-1), monoRight(-1)
{
    if (previous && previous->HasVelocity())
    {
        SetVelocity(previous->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }
}

void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;

    int nReserve = 0.5f * N / (nCells);

    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
        {
            mGrid[i][j].reserve(nReserve);
            if (is_fisheye_stereo)
            {
                mGridRight[i][j].reserve(nReserve);
            }
        }
    }

    for (int i = 0; i < N; i++)
    {
        const auto &kp = (!is_fisheye_stereo || std::cmp_less(i, mvKeys.size())) ? mvKeys[i] : mvKeysRight[i - static_cast<int>(mvKeys.size())];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp.undistorted(), nGridPosX, nGridPosY))
        {
            if (!is_fisheye_stereo || std::cmp_less(i, mvKeys.size()))
            {
                mGrid[nGridPosX][nGridPosY].push_back(i);
            }
            else
            {
                mGridRight[nGridPosX][nGridPosY].push_back(i - static_cast<int>(mvKeys.size()));
            }
        }
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

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw           = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

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

Sophus::SE3<float> Frame::GetImuPose() { return mTcw.inverse() * mImuCalib.mTcb; }

Sophus::SE3f Frame::GetRelativePoseTrl() { return mTrl; }

Sophus::SE3f Frame::GetRelativePoseTlr() { return mTlr; }

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation() { return mTlr.rotationMatrix(); }

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() { return mTlr.translation(); }

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit, const camera_vector &cameras)
{
    if (!is_fisheye_stereo)
    {
        pMP->mbTrackInView = false;
        pMP->mTrackProjX   = -1;
        pMP->mTrackProjY   = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float, 3, 1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float, 3, 1> Pc      = mRcw * P + mtcw;
        const float                      Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ  = Pc(2);
        const float  invz = 1.0f / PcZ;
        if (PcZ < 0.0f)
        {
            return false;
        }

        const Eigen::Vector2f uv = cameras[0]->project(Pc);

        if (uv(0) < mnMinX || uv(0) > mnMaxX)
        {
            return false;
        }
        if (uv(1) < mnMinY || uv(1) > mnMaxY)
        {
            return false;
        }

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

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
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
        {
            return false;
        }

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX   = uv(0);
        pMP->mTrackProjXR  = uv(0) - mbf * invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY       = uv(1);
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos     = viewCos;

        return true;
    }
    else
    {
        pMP->mbTrackInView      = false;
        pMP->mbTrackInViewR     = false;
        pMP->mnTrackScaleLevel  = -1;
        pMP->mnTrackScaleLevelR = -1;

        pMP->mbTrackInView  = isInFrustumChecks(pMP, viewingCosLimit, *cameras[0]);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP, viewingCosLimit, *cameras[1], true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint *pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

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
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw) { return mRcw * pCw + mtcw; }

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

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
                const auto &kpUn = (!is_fisheye_stereo || !bRight) ? mvKeys[vCell[j]] : mvKeysRight[vCell[j]];
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

                const float distx = kpUn.pt_undistorted.x - x;
                const float disty = kpUn.pt_undistorted.y - y;

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
        auto       vCurrentDesc      = place_descriptor_vector(mvKeys);
        const auto right_descriptors = place_descriptor_vector(mvKeysRight);
        vCurrentDesc.insert(vCurrentDesc.end(), right_descriptors.begin(), right_descriptors.end());
        place_vocabulary_ptr->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::UndistortKeyPoints()
{
    if (mDistCoef.at<float>(0) == 0.0)
    {
        for (auto &key : mvKeys)
        {
            key.pt_undistorted = key.pt;
        }
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

    for (int i = 0; i < N; i++)
    {
        mvKeys[i].pt_undistorted.x = mat.at<float>(i, 0);
        mvKeys[i].pt_undistorted.y = mat.at<float>(i, 1);
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

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

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

void Frame::ComputeStereoMatches(const std::vector<cv::Mat> &left_pyramid, const std::vector<cv::Mat> &right_pyramid)
{
    mvuRight = vector<float>(N, -1.0f);
    mvDepth  = vector<float>(N, -1.0f);

    const int descriptor_threshold = (descriptor_matcher::high_distance_threshold + descriptor_matcher::low_distance_threshold) / 2;

    const int nRows = left_pyramid[0].rows;

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
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf / minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int>> vDistIdx;
    vDistIdx.reserve(N);

    for (int iL = 0; iL < N; iL++)
    {
        const cv::KeyPoint &kpL    = mvKeys[iL];
        const int          &levelL = kpL.octave;
        const float        &vL     = kpL.pt.y;
        const float        &uL     = kpL.pt.x;

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

        float  bestDist = descriptor_matcher::high_distance_threshold;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mvKeys[iL].descriptor;

        // Compare descriptor to right keypoints
        const auto ranked_candidates = descriptor_matcher::rank_candidates(dL, mvKeysRight, vCandidates);
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
                const cv::Mat &dR   = mvKeysRight[iR].descriptor;
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
            const float uR0         = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL    = round(kpL.pt.x * scaleFactor);
            const float scaledvL    = round(kpL.pt.y * scaleFactor);
            const float scaleduR0   = round(uR0 * scaleFactor);

            // sliding window search
            const int w  = 5;
            cv::Mat   IL = left_pyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);

            float         bestDist = INT_MAX;
            int           bestincR = 0;
            const int     L        = 5;
            vector<float> vDists;
            vDists.resize(2 * L + 1);

            const float iniu = scaleduR0 + L - w;
            const float endu = scaleduR0 + L + w + 1;
            if (iniu < 0 || endu >= right_pyramid[kpL.octave].cols)
            {
                continue;
            }

            for (int incR = -L; incR <= +L; incR++)
            {
                cv::Mat IR = right_pyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

                float dist = cv::norm(IL, IR, cv::NORM_L1);
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
            const float dist1 = vDists[L + bestincR - 1];
            const float dist2 = vDists[L + bestincR];
            const float dist3 = vDists[L + bestincR + 1];

            const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

            if (deltaR < -1 || deltaR > 1)
            {
                continue;
            }

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

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
        const auto &kp = mvKeys[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v, u);

        if (d > 0)
        {
            mvDepth[i]  = d;
            mvuRight[i] = kp.pt_undistorted.x - mbf / d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if (z > 0)
    {
        const float     u = mvKeys[i].pt_undistorted.x;
        const float     v = mvKeys[i].pt_undistorted.y;
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

void Frame::ComputeStereoFishEyeMatches(const camera_vector &cameras)
{
    cv::Mat stereoDescLeft  = descriptor_matrix(mvKeys, monoLeft);
    cv::Mat stereoDescRight = descriptor_matrix(mvKeysRight, monoRight);

    mvLeftToRightMatch = vector<int>(mvKeys.size(), -1);
    mvRightToLeftMatch = vector<int>(mvKeysRight.size(), -1);
    mvDepth            = vector<float>(mvKeys.size(), -1.0f);
    mvuRight           = vector<float>(mvKeys.size(), -1);
    mvStereo3Dpoints   = vector<Eigen::Vector3f>(mvKeys.size());
    mnCloseMPs         = 0;

    // Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    descriptor_matcher::knn_match(stereoDescLeft, stereoDescRight, matches, 2);

    cv::parallel_for_(
        cv::Range(0, static_cast<int>(matches.size())),
        [&](const cv::Range &range)
        {
            for (int index = range.start; index < range.end; ++index)
            {
                const auto &neighbors = matches[index];
                if (neighbors.size() < 2 || !(neighbors[0].distance < neighbors[1].distance * 0.7))
                {
                    continue;
                }
                const auto      left_index  = neighbors[0].queryIdx + monoLeft;
                const auto      right_index = neighbors[0].trainIdx + monoRight;
                const auto      sigma_left  = mvLevelSigma2[mvKeys[left_index].octave];
                const auto      sigma_right = mvLevelSigma2[mvKeysRight[right_index].octave];
                Eigen::Vector3f point;
                const auto      depth =
                    static_cast<KannalaBrandt8 &>(*cameras[0]).TriangulateMatches(*cameras[1], mvKeys[left_index], mvKeysRight[right_index], mRlr, mtlr, sigma_left, sigma_right, point);
                if (depth > 0.0001f)
                {
                    mvLeftToRightMatch[left_index] = right_index;
                    mvStereo3Dpoints[left_index]   = point;
                    mvDepth[left_index]            = depth;
                }
            }
        },
        matches.size() >= 128 ? 4 : 1);
    for (const auto &neighbors : matches)
    {
        if (!neighbors.empty())
        {
            const auto left_index  = neighbors[0].queryIdx + monoLeft;
            const auto right_index = mvLeftToRightMatch[left_index];
            if (right_index >= 0)
            {
                mvRightToLeftMatch[right_index] = left_index;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, GeometricCamera &camera, bool bRight)
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
    uv = camera.project(Pc);

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
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

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
