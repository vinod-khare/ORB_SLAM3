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

#include "Tracking.h"

#include "Converter.h"
#include "G2oTypes.h"
#include "GeometricTools.h"
#include "KannalaBrandt8.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "descriptor_matcher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

#include <opencv2/video/tracking.hpp>

#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>

using namespace std;

namespace ORB_SLAM3
{

Tracking::Tracking(System *pSys, place_vocabulary *pVoc, Atlas *pAtlas, KeyFrameDatabase *pKFDB, const int sensor, Settings *settings, const string &_nameSeq)
    : mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false), mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), place_vocabulary_ptr(pVoc), mpKeyFrameDB(pKFDB),
      mbReadyToInitializate(false), mpSystem(pSys), bStepByStep(false), mpAtlas(pAtlas), mpLastKeyFrame(nullptr), mnLastRelocFrameId(0), time_recently_lost(5.0), mnFirstFrameId(0),
      mnInitialFrameId(0), mbCreatedMap(false), mpCamera2(nullptr)
{
    newParameterLoader(settings);

    initID                           = 0;
    lastID                           = 0;
    mbInitWith3KFs                   = false;
    mnNumDataset                     = 0;

    vector<GeometricCamera *> vpCams = mpAtlas->GetAllCameras();
    spdlog::debug("There are {} cameras in the atlas", vpCams.size());
    for (GeometricCamera *pCam : vpCams)
    {
        const char *camera_type = "unknown";
        if (pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            camera_type = "pinhole";
        }
        else if (pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            camera_type = "fisheye";
        }
        spdlog::debug("Camera {} is {}", pCam->GetId(), camera_type);
    }
}

Tracking::~Tracking() {}

// TODO: don't use raw pointer
void Tracking::newParameterLoader(Settings *settings)
{
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if (settings->needToUndistort())
    {
        mDistCoef = settings->camera1DistortionCoef();
    }
    else
    {
        mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    }

    // TODO: missing image scaling and rectification
    mImageScale        = 1.0f;

    mK                 = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0, 0) = mpCamera->getParameter(0);
    mK.at<float>(1, 1) = mpCamera->getParameter(1);
    mK.at<float>(0, 2) = mpCamera->getParameter(2);
    mK.at<float>(1, 2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0, 0) = mpCamera->getParameter(0);
    mK_(1, 1) = mpCamera->getParameter(1);
    mK_(0, 2) = mpCamera->getParameter(2);
    mK_(1, 2) = mpCamera->getParameter(3);

    if ((mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && settings->cameraType() == Settings::KannalaBrandt)
    {
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr      = settings->Tlr();
    }

    if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mbf      = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
    {
        mDepthMapFactor = settings->depthMapFactor();
        if (fabs(mDepthMapFactor) < 1e-5)
        {
            mDepthMapFactor = 1;
        }
        else
        {
            mDepthMapFactor = 1.0f / mDepthMapFactor;
        }
    }

    mMinFrames                                = 0;
    mMaxFrames                                = settings->fps();
    mnFramesToResetIMU                        = mMaxFrames;
    mbRGB                                     = settings->rgb();

    // ORB parameters
    int                          nFeatures    = settings->nFeatures();
    int                          nLevels      = settings->nLevels();
    int                          fIniThFAST   = settings->initThFAST();
    int                          fMinThFAST   = settings->minThFAST();
    float                        fScaleFactor = settings->scaleFactor();
    const keypoint_detector_type detector     = settings->keypoint_detector();
    const auto                   descriptor   = settings->keypoint_descriptor();
    descriptor_matcher::set_backend(settings->matcher_backend());
    _use_optical_flow = settings->use_optical_flow() && mSensor != System::RGBD && mSensor != System::IMU_RGBD;

    spdlog::info("Sparse optical flow: {}", _use_optical_flow ? "enabled" : "disabled");

    left_keypoint_extractor = std::make_unique<keypoint_extractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);

    if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
    {
        right_keypoint_extractor = std::make_unique<keypoint_extractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);
    }

    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        initial_keypoint_extractor = std::make_unique<keypoint_extractor>(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);
    }

    // IMU parameters
    Sophus::SE3f Tbc             = settings->Tbc();
    mInsertKFsLost               = settings->insertKFsWhenLost();
    mImuFreq                     = settings->imuFrequency();
    mImuPer                      = 0.001; // TODO: ESTO ESTA BIEN?
    float       Ng               = settings->noiseGyro();
    float       Na               = settings->noiseAcc();
    float       Ngw              = settings->gyroWalk();
    float       Naw              = settings->accWalk();

    const float sf               = sqrt(mImuFreq);
    mpImuCalib                   = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) { mpLocalMapper = pLocalMapper; }

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) { mpLoopClosing = pLoopClosing; }

void Tracking::SetResetHooks(std::function<void()> on_reset_start, std::function<void()> on_reset_complete)
{
    mOnResetStart    = std::move(on_reset_start);
    mOnResetComplete = std::move(on_reset_complete);
}

void         Tracking::SetStepByStep(bool bSet) { bStepByStep = bSet; }

bool         Tracking::GetStepByStep() { return bStepByStep; }

Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{

    mImGray             = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight            = imRectRight;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGRA2GRAY);
        }
    }

    mImRight = imGrayRight;
    // TODO: why does Frame class need the extractor object? I think the extractor can be external to the frame. This will reduce coupling and potentially improve performance.
    if (can_defer_features(timestamp))
    {
        mCurrentFrame = Frame(mLastFrame, timestamp);
    }
    else if (mSensor == System::STEREO && !mpCamera2)
    {
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, left_keypoint_extractor.get(), right_keypoint_extractor.get(), place_vocabulary_ptr, mK, mDistCoef, mbf, mThDepth, mpCamera);
    }
    else if (mSensor == System::STEREO && mpCamera2)
    {
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, left_keypoint_extractor.get(), right_keypoint_extractor.get(), place_vocabulary_ptr, mK, mDistCoef, mbf, mThDepth, mpCamera,
                              mpCamera2, mTlr);
    }
    else if (mSensor == System::IMU_STEREO && !mpCamera2)
    {
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, left_keypoint_extractor.get(), right_keypoint_extractor.get(), place_vocabulary_ptr, mK, mDistCoef, mbf, mThDepth, mpCamera,
                              &mLastFrame, *mpImuCalib);
    }
    else if (mSensor == System::IMU_STEREO && mpCamera2)
    {
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, left_keypoint_extractor.get(), right_keypoint_extractor.get(), place_vocabulary_ptr, mK, mDistCoef, mbf, mThDepth, mpCamera,
                              mpCamera2, mTlr, &mLastFrame, *mpImuCalib);
    }

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    mCurrentFrame.optical_flow_pyramid.swap(next_optical_flow_pyramid);
    if (!mCurrentFrame.flow_only)
    {
        last_feature_frame_id_ = mCurrentFrame.mnId;
        ++feature_frames_;
    }
    Track();
    if (mCurrentFrame.flow_only && mState == OK)
    {
        ++flow_frames_;
    }

    if (_use_optical_flow && mLastFrame.mnId == mCurrentFrame.mnId && !mImGray.empty())
    {
        _previous_gray          = mImGray.clone();
        _previous_gray_frame_id = mCurrentFrame.mnId;
    }
    else
    {
        _previous_gray.release();
    }

    return mCurrentFrame.GetPose();
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray         = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
        }
    }

    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
    {
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);
    }

    if (mSensor == System::RGBD)
    {
        mCurrentFrame = Frame(mImGray, imDepth, timestamp, left_keypoint_extractor.get(), place_vocabulary_ptr, mK, mDistCoef, mbf, mThDepth);
    }
    else if (mSensor == System::IMU_RGBD)
    {
        mCurrentFrame = Frame(mImGray, imDepth, timestamp, left_keypoint_extractor.get(), place_vocabulary_ptr, mK, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
    }

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    Track();

    return mCurrentFrame.GetPose();
}

Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;

    // Save color image (BGR) for display before grayscale conversion
    if (!mNextFrameColor.empty())
    {
        mImColor        = std::move(mNextFrameColor);
        mNextFrameColor = cv::Mat();
    }
    else if (im.channels() == 3)
    {
        if (mbRGB)
        {
            cv::cvtColor(im, mImColor, cv::COLOR_RGB2BGR);
        }
        else
        {
            mImColor = im.clone();
        }
    }
    else if (im.channels() == 4)
    {
        if (mbRGB)
        {
            cv::cvtColor(im, mImColor, cv::COLOR_RGBA2BGR);
        }
        else
        {
            cv::cvtColor(im, mImColor, cv::COLOR_BGRA2BGR);
        }
    }
    else
    {
        cv::cvtColor(im, mImColor, cv::COLOR_GRAY2BGR);
    }

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
        }
    }

    if (can_defer_features(timestamp))
    {
        mCurrentFrame = Frame(mLastFrame, timestamp);
    }
    else if (mSensor == System::MONOCULAR)
    {
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames)
        {
            mCurrentFrame = Frame(mImGray, timestamp, initial_keypoint_extractor.get(), place_vocabulary_ptr, mpCamera, mDistCoef, mbf, mThDepth);
        }
        else
        {
            mCurrentFrame = Frame(mImGray, timestamp, left_keypoint_extractor.get(), place_vocabulary_ptr, mpCamera, mDistCoef, mbf, mThDepth);
        }
    }
    else if (mSensor == System::IMU_MONOCULAR)
    {
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray, timestamp, initial_keypoint_extractor.get(), place_vocabulary_ptr, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
        }
        else
        {
            mCurrentFrame = Frame(mImGray, timestamp, left_keypoint_extractor.get(), place_vocabulary_ptr, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
        }
    }

    if (mState == NO_IMAGES_YET)
    {
        t0 = timestamp;
    }

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    lastID                  = mCurrentFrame.mnId;
    mCurrentFrame.optical_flow_pyramid.swap(next_optical_flow_pyramid);
    if (!mCurrentFrame.flow_only)
    {
        last_feature_frame_id_ = mCurrentFrame.mnId;
        ++feature_frames_;
    }
    Track();
    if (mCurrentFrame.flow_only && mState == OK)
    {
        ++flow_frames_;
    }

    if (_use_optical_flow && mLastFrame.mnId == mCurrentFrame.mnId && !mImGray.empty())
    {
        _previous_gray          = mImGray.clone();
        _previous_gray_frame_id = mCurrentFrame.mnId;
    }
    else
    {
        _previous_gray.release();
    }

    return mCurrentFrame.GetPose();
}

void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if (!mCurrentFrame.mpPrevFrame)
    {
        spdlog::debug("non prev frame");
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if (mlQueueImuData.size() == 0)
    {
        spdlog::debug("Not IMU data in mlQueueImuData!!");
        mCurrentFrame.setIntegrated();
        return;
    }

    while (true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if (!mlQueueImuData.empty())
            {
                IMU::Point *m = &mlQueueImuData.front();
                if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if (m->t < mCurrentFrame.mTimeStamp - mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if (bSleep)
        {
            usleep(500);
        }
    }

    const int n = mvImuFromLastFrame.size() - 1;
    if (n == 0)
    {
        spdlog::warn("Empty IMU measurements vector");
        return;
    }

    IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);

    for (int i = 0; i < n; i++)
    {
        float           tstep;
        Eigen::Vector3f acc, angVel;
        if ((i == 0) && (i < (n - 1)))
        {
            float tab  = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc        = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
            angVel     = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
            tstep      = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if (i < (n - 1))
        {
            acc    = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
            tstep  = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
        }
        else if ((i > 0) && (i == (n - 1)))
        {
            float tab  = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
            acc        = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
            angVel     = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
            tstep      = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
        }
        else if ((i == 0) && (i == (n - 1)))
        {
            acc    = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep  = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
        {
            spdlog::error("IMU preintegration from the last keyframe is missing");
        }
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated      = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame          = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();
}

bool Tracking::PredictStateIMU()
{
    if (!mCurrentFrame.mpPrevFrame)
    {
        spdlog::trace("No last frame");
        return false;
    }

    if (mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float           t12  = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f       Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f       twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f       Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

        mCurrentFrame.mImuBias  = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if (!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float           t12  = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f       Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f       twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f       Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

        mCurrentFrame.mImuBias  = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
    {
        spdlog::warn("IMU prediction unavailable");
    }

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}

void Tracking::Track()
{

    if (bStepByStep)
    {
        spdlog::debug("Tracking is waiting for the next step");
        while (!mbStep && bStepByStep)
        {
            usleep(500);
        }
        mbStep = false;
    }

    if (mpLocalMapper->mbBadImu)
    {
        spdlog::warn("Resetting map because local mapper set the bad IMU flag");
        mpSystem->ResetActiveMap();
        return;
    }

    Map *pCurrentMap = mpAtlas->GetCurrentMap();
    if (!pCurrentMap)
    {
        spdlog::error("There is no active map in the Atlas");
    }

    if (mState != NO_IMAGES_YET)
    {
        if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp)
        {
            spdlog::error("Frame with a timestamp older than the previous frame detected");
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if (mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0)
        {
            if (mpAtlas->isInertial())
            {

                if (mpAtlas->isImuInitialized())
                {
                    spdlog::warn("Timestamp jump detected; setting state to LOST and resetting IMU integration");
                    if (!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    spdlog::warn("Timestamp jump detected before IMU initialization; resetting map");
                    mpSystem->ResetActiveMap();
                }
                return;
            }
        }
    }

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
    {
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
    }

    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
        PreintegrateIMU();
    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated           = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex    = pCurrentMap->GetLastMapChange();
    if (nCurMapChangeIndex > nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }

    if (mState == NOT_INITIALIZED)
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization();
        }

        if (mState != OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if (mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK = false;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if (mState == OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if ((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    spdlog::trace("TRACK: Track with respect to the reference KF");
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    spdlog::trace("TRACK: Track with motion model");
                    bOK = TrackWithMotionModel();
                    if (!bOK)
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }

                if (!bOK)
                {
                    if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if (pCurrentMap->KeyFramesInMap() > 10)
                    {
                        mState         = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST)
                {
                    spdlog::trace("Lost for a short time");

                    bOK = true;
                    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        if (pCurrentMap->isImuInitialized())
                        {
                            PredictStateIMU();
                        }
                        else
                        {
                            bOK = false;
                        }

                        if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost)
                        {
                            mState = LOST;
                            spdlog::debug("Track Lost...");
                            bOK = false;
                        }
                    }
                    else
                    {
                        // Relocalization
                        bOK = Relocalization();
                        if (mCurrentFrame.mTimeStamp - mTimeStampLost > 3.0f && !bOK)
                        {
                            mState = LOST;
                            spdlog::debug("Track Lost...");
                            bOK = false;
                        }
                    }
                }
                else if (mState == LOST)
                {

                    spdlog::debug("A new map is started...");

                    if (pCurrentMap->KeyFramesInMap() < 10)
                    {
                        mpSystem->ResetActiveMap();
                        spdlog::debug("Reseting current map...");
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }

                    if (mpLastKeyFrame)
                    {
                        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
                    }

                    spdlog::debug("done");

                    return;
                }
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if (mState == LOST)
            {
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                {
                    spdlog::debug("IMU. State LOST");
                }
                bOK = Relocalization();
            }
            else
            {
                if (!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if (mbVelocity)
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool               bOKMM    = false;
                    bool               bOKReloc = false;
                    vector<MapPoint *> vpMPsMM;
                    vector<bool>       vbOutMM;
                    Sophus::SE3f       TcwMM;
                    if (mbVelocity)
                    {
                        bOKMM   = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM   = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier   = vbOutMM;

                        if (mbVO)
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if (bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
        {
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (!mbOnlyTracking)
        {
            if (bOK)
            {
                bOK = TrackLocalMap();
            }
            if (!bOK)
            {
                spdlog::warn("Failed to track local map");
            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if (bOK && !mbVO)
            {
                bOK = TrackLocalMap();
            }
        }

        if (mCurrentFrame.flow_only && !bOK)
        {
            ++flow_recoveries_;
            spdlog::debug("Frame {}: extracting features for local-map recovery", mCurrentFrame.mnId);
            extract_current_features();
            bOK = TrackWithMotionModel();
            if (bOK)
            {
                bOK = TrackLocalMap();
            }
        }

        if (bOK)
        {
            mState = OK;
        }
        else if (mState == OK)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                spdlog::debug("Track lost for less than one second...");
                if (!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    spdlog::warn("IMU is not or recently initialized; resetting active map");
                    mpSystem->ResetActiveMap();
                }

                mState = RECENTLY_LOST;
            }
            else
            {
                mState = RECENTLY_LOST; // visual to lost
            }

            mTimeStampLost = mCurrentFrame.mTimeStamp;
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if ((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) && (mCurrentFrame.mnId > static_cast<unsigned int>(mnFramesToResetIMU)) &&
            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            spdlog::debug("Saving pointer to frame. imu needs reset...");
            Frame *pF                   = new Frame(mCurrentFrame);
            pF->mpPrevFrame             = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if (pCurrentMap->isImuInitialized())
        {
            if (bOK)
            {
                if (mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU))
                {
                    spdlog::debug("Resetting frame");
                    ResetFrameIMU();
                }
                else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30))
                {
                    mLastBias = mCurrentFrame.mImuBias;
                }
            }
        }

        if (bOK || mState == RECENTLY_LOST)
        {
            // Update motion model
            if (mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity            = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity           = true;
            }
            else
            {
                mbVelocity = false;
            }

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i]   = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
                }
            }

            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            if (bNeedKF && (bOK || (mInsertKFsLost && mState == RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
            {
                CreateNewKeyFrame();
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (pCurrentMap->KeyFramesInMap() <= 10)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                if (!pCurrentMap->isImuInitialized())
                {
                    spdlog::warn("Track lost before IMU initialisation; resetting");
                    mpSystem->ResetActiveMap();
                    return;
                }
            }

            CreateMapInAtlas();

            return;
        }

        if (!mCurrentFrame.mpReferenceKF)
        {
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }

        mLastFrame = Frame(mCurrentFrame);
    }

    if (mState == OK || mState == RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }
    }

#ifdef REGISTER_LOOP
    if (Stop())
    {

        // Safe area to stop
        while (isStopped())
        {
            usleep(3000);
        }
    }
#endif
}

void Tracking::StereoInitialization()
{
    if (mCurrentFrame.N > 500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                spdlog::debug("No IMU measurements");
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA - mLastFrame.mpImuPreintegratedFrame->avgA).norm() < 0.5)
            {
                spdlog::debug("Insufficient acceleration for initialization");
                return;
            }

            if (mpImuPreintegratedFromLastKF)
            {
                delete mpImuPreintegratedFromLastKF;
            }

            mpImuPreintegratedFromLastKF     = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
        {
            mCurrentFrame.SetPose(Sophus::SE3f());
        }

        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB, mpCamera, mpCamera2);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if (!mpCamera2)
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }
        }
        else
        {
            for (int i = 0; i < mCurrentFrame.Nleft; i++)
            {
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if (rightIndex != -1)
                {
                    Eigen::Vector3f x3D    = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint       *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini, i);
                    pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP, i);
                    pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]                                = pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
                }
            }
        }

        spdlog::info("New Map created with {} points", mpAtlas->MapPointsInMap());

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame       = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame   = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints           = mpAtlas->GetAllMapPoints();
        mpReferenceKF               = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
    }
}

void Tracking::MonocularInitialization()
{

    if (!mbReadyToInitializate)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame    = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
            {
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
            }

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if (mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF     = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.mvKeys.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0)))
        {
            mbReadyToInitializate = false;

            return;
        }

        // Find correspondences
        descriptor_matcher matcher(0.9, true);
        int                nmatches = matcher.search_for_initialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches < 100)
        {
            mbReadyToInitializate = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB, mpCamera, mpCamera2);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB, mpCamera, mpCamera2);

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);
    }

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
        {
            continue;
        }

        // Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]]   = false;

        // Add to Map
        mpAtlas->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint *> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    spdlog::info("New Map created with {} points", mpAtlas->MapPointsInMap());
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if (mSensor == System::IMU_MONOCULAR)
    {
        invMedianDepth = 4.0f / medianDepth; // 4.0f
    }
    else
    {
        invMedianDepth = 1.0f / medianDepth;
    }

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
    {
        spdlog::warn("Wrong initialization; resetting");
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w   = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF              = pKFini;
        pKFini->mNextKF              = pKFcur;
        pKFcur->mpImuPreintegrated   = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints            = mpAtlas->GetAllMapPoints();
    mpReferenceKF                = pKFcur;
    mCurrentFrame.mpReferenceKF  = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame *> vKFs      = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f       deltaT    = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity                   = false;
    Eigen::Vector3f phi          = deltaT.so3().log();

    double          aux          = (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) / (mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
    phi                         *= aux;

    mLastFrame                   = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;

    initID = pKFcur->mnId;
}

void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
    {
        mpAtlas->SetInertialSensor();
    }
    mbSetInit        = false;

    mnInitialFrameId = mCurrentFrame.mnId + 1;
    mState           = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity       = false;
    spdlog::debug("First frame id in map: {}", mnLastInitFrameId + 1);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
    }

    if (mpLastKeyFrame)
    {
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    }

    if (mpReferenceKF)
    {
        mpReferenceKF = static_cast<KeyFrame *>(NULL);
    }

    mLastFrame    = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    extract_current_features();
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    descriptor_matcher matcher(0.7, true);
    vector<MapPoint *> vpMapPointMatches;

    int                nmatches = matcher.search_by_vocabulary(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        spdlog::debug("Reference keyframe tracking found fewer than 15 matches");
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP                 = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i]   = false;
                if (i < mCurrentFrame.Nleft)
                {
                    pMP->mbTrackInView = false;
                }
                else
                {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView   = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            {
                nmatchesMap++;
            }
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        return true;
    }
    else
    {
        return nmatchesMap >= 10;
    }
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame    *pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr  = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking)
    {
        return;
    }

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int>> vDepthIdx;
    const int                Nfeat = mLastFrame.Nleft == -1 ? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for (int i = 0; i < Nfeat; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
    {
        return;
    }

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        int       i          = vDepthIdx[j].second;

        bool      bCreateNew = false;

        MapPoint *pMP        = mLastFrame.mvpMapPoints[i];

        if (!pMP)
        {
            bCreateNew = true;
        }
        else if (pMP->Observations() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            Eigen::Vector3f x3D;

            if (mLastFrame.Nleft == -1)
            {
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else
            {
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint *pNewMP           = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);
            mLastFrame.mvpMapPoints[i] = pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
        {
            break;
        }
    }
}

bool Tracking::TrackWithMotionModel()
{
    descriptor_matcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    const bool imu_prediction = mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU);

    if (imu_prediction)
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }

    if (mCurrentFrame.flow_only)
    {
        const auto predicted_pose = mCurrentFrame.GetPose();
        const int  flow_matches   = track_with_optical_flow();
        if (flow_matches >= 50)
        {
            Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);
            if (mCurrentFrame.reliable_flow_inliers(mImGray.size()))
            {
                spdlog::debug("Frame {}: flow-only tracking, {} observations", mCurrentFrame.mnId, flow_matches);
                return true;
            }
        }
        spdlog::debug("Frame {}: optical flow weak; extracting features for descriptor recovery", mCurrentFrame.mnId);
        ++flow_recoveries_;
        extract_current_features();
        mCurrentFrame.SetPose(predicted_pose);
    }

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    if (imu_prediction)
    {
        return true;
    }

    // Project points seen in previous frame
    int th;

    if (mSensor == System::STEREO)
    {
        th = 7;
    }
    else
    {
        th = 15;
    }

    // TODO: projecting points should be trackers responsiblity, leaving matcher to match pre-projected points.
    int nmatches = matcher.search_by_projection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR, mpCamera);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        spdlog::debug("Not enough matches, wider window search!!");
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        nmatches = matcher.search_by_projection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR, mpCamera);
        spdlog::debug("Matches with wider search: " + to_string(nmatches));
    }

    if (nmatches < 20)
    {
        spdlog::debug("Not enough matches!!");
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP                 = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i]   = false;
                if (i < mCurrentFrame.Nleft)
                {
                    pMP->mbTrackInView = false;
                }
                else
                {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            {
                nmatchesMap++;
            }
        }
    }

    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        return true;
    }
    else
    {
        return nmatchesMap >= 10;
    }
}

bool Tracking::can_defer_features(double timestamp) const
{
    // Localization-only mode also needs fresh stereo depth for temporary VO points.
    return _use_optical_flow && !mbOnlyTracking && mState == OK && mbVelocity && !Frame::mbInitialComputations && Frame::nNextId > mnLastRelocFrameId + 2 &&
           Frame::nNextId - last_feature_frame_id_ < static_cast<unsigned long>(std::max(2, mMaxFrames / 4)) && timestamp > mLastFrame.mTimeStamp && timestamp - mLastFrame.mTimeStamp < 1.0 &&
           !_previous_gray.empty() && _previous_gray_frame_id == mLastFrame.mnId && _previous_gray.size() == mImGray.size() && mImGray.type() == CV_8UC1;
}

void Tracking::extract_current_features()
{
    if (!mCurrentFrame.flow_only)
    {
        return;
    }
    mCurrentFrame.extract_features(mImGray, mpCamera, (mSensor == System::STEREO || mSensor == System::IMU_STEREO) ? mImRight : cv::Mat(), mpCamera2);
    last_feature_frame_id_ = mCurrentFrame.mnId;
    ++feature_frames_;
}

void Tracking::report_flow_statistics() const
{
    if (_use_optical_flow)
    {
        spdlog::info("Optical flow: {} flow-only frames, {} feature-extracted frames, {} recovery attempts", flow_frames_, feature_frames_, flow_recoveries_);
    }
}

int Tracking::track_with_optical_flow()
{
    flow_reference_matches_ = 0;
    if (!_use_optical_flow || _previous_gray.empty() || mImGray.empty() || _previous_gray_frame_id != mLastFrame.mnId || _previous_gray.size() != mImGray.size() || mImGray.type() != CV_8UC1)
    {
        return 0;
    }

    const std::size_t previous_left_count = mLastFrame.mvKeys.size();
    const std::size_t current_left_count  = mCurrentFrame.mvKeys.size();

    if ((mLastFrame.Nleft >= 0 && previous_left_count != static_cast<std::size_t>(mLastFrame.Nleft)) ||
        (mCurrentFrame.Nleft >= 0 && current_left_count != static_cast<std::size_t>(mCurrentFrame.Nleft)) ||
        (mLastFrame.Nleft < 0 && previous_left_count != static_cast<std::size_t>(mLastFrame.N)) ||
        (mCurrentFrame.Nleft < 0 && current_left_count != static_cast<std::size_t>(mCurrentFrame.N)) || mLastFrame.N != static_cast<int>(mLastFrame.mvpMapPoints.size()) ||
        mLastFrame.N != static_cast<int>(mLastFrame.mvbOutlier.size()) || mCurrentFrame.N != static_cast<int>(mCurrentFrame.mvpMapPoints.size()))
    {
        return 0;
    }

    std::vector<cv::Point2f>       previous_points;
    std::vector<cv::Point2f>       current_points;
    std::vector<cv::KeyPoint>      previous_keys;
    std::vector<cv::KeyPoint>      previous_display_keys;
    std::vector<MapPoint *>        map_points;
    std::array<int, 16>            cell_counts{};
    std::unordered_set<MapPoint *> selected_points;
    CV_Assert(mLastFrame.display_keys.size() == mLastFrame.mvKeys.size());
    previous_points.reserve(mLastFrame.mvKeys.size());
    map_points.reserve(mLastFrame.mvKeys.size());

    for (std::size_t index = 0; index < mLastFrame.mvKeys.size(); ++index)
    {
        MapPoint *map_point = mLastFrame.mvpMapPoints[index];

        if (!map_point || mLastFrame.mvbOutlier[index] || map_point->isBad() || map_point->Observations() == 0)
        {
            continue;
        }

        const auto &key = mLastFrame.mvKeys[index];
        if (!std::isfinite(key.pt.x) || !std::isfinite(key.pt.y) || key.pt.x < 0 || key.pt.y < 0 || key.pt.x >= mImGray.cols || key.pt.y >= mImGray.rows)
        {
            continue;
        }
        const int x          = static_cast<int>(key.pt.x * 4 / mImGray.cols);
        const int y          = static_cast<int>(key.pt.y * 4 / mImGray.rows);
        auto     &cell_count = cell_counts[static_cast<std::size_t>(y * 4 + x)];
        if (cell_count >= 40 || !selected_points.insert(map_point).second)
        {
            continue;
        }
        ++cell_count;
        previous_points.push_back(key.pt);
        previous_keys.push_back(key);
        previous_display_keys.push_back(mLastFrame.display_keys[index]);
        map_points.push_back(map_point);
        cv::Point2f prediction = key.pt;
        // Camera-model projections are in LK coordinates except for distorted pinhole input.
        if (mpCamera2 || mDistCoef.at<float>(0) == 0.0f)
        {
            const Eigen::Vector3f position = mCurrentFrame.GetPose() * map_point->GetWorldPos();
            if (position.z() > 0)
            {
                const Eigen::Vector2f projected = mpCamera->project(position);
                if (projected.allFinite() && projected.x() >= 0 && projected.y() >= 0 && projected.x() < mImGray.cols && projected.y() < mImGray.rows)
                {
                    prediction = cv::Point2f(projected.x(), projected.y());
                }
            }
        }
        current_points.push_back(prediction);
    }

    if (previous_points.empty())
    {
        return 0;
    }
    flow_reference_matches_ = static_cast<int>(previous_points.size());

    std::vector<unsigned char> forward_status;
    std::vector<float>         forward_error;
    const cv::Size             window_size(21, 21);
    const cv::TermCriteria     criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    const auto                &previous_pyramid = mLastFrame.get_optical_flow_pyramid(_previous_gray);
    const auto                &current_pyramid  = mCurrentFrame.get_optical_flow_pyramid(mImGray);
    cv::calcOpticalFlowPyrLK(previous_pyramid, current_pyramid, previous_points, current_points, forward_status, forward_error, window_size, 3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<cv::Point2f>   backward_points;
    std::vector<unsigned char> backward_status;
    std::vector<float>         backward_error;
    cv::calcOpticalFlowPyrLK(current_pyramid, previous_pyramid, current_points, backward_points, backward_status, backward_error, window_size, 3, criteria);

    std::vector<cv::KeyPoint> tracked_keys;
    std::vector<cv::KeyPoint> tracked_display_keys;
    std::vector<MapPoint *>   tracked_points;
    tracked_keys.reserve(previous_points.size());
    tracked_points.reserve(previous_points.size());

    for (std::size_t index = 0; index < previous_points.size(); ++index)
    {
        if (!forward_status[index] || !backward_status[index] || !std::isfinite(forward_error[index]))
        {
            continue;
        }

        const float delta_x           = previous_points[index].x - backward_points[index].x;
        const float delta_y           = previous_points[index].y - backward_points[index].y;
        const float consistency_error = std::sqrt(delta_x * delta_x + delta_y * delta_y);

        if (!std::isfinite(consistency_error) || consistency_error > 1.5f || forward_error[index] > 50.0f)
        {
            continue;
        }

        const cv::Point2f &point = current_points[index];

        if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0.0f || point.x >= mImGray.cols || point.y < 0.0f || point.y >= mImGray.rows)
        {
            continue;
        }

        auto key = previous_keys[index];
        key.pt   = point;
        tracked_keys.push_back(key);
        auto display_key = previous_display_keys[index];
        display_key.pt   = point;
        tracked_display_keys.push_back(display_key);
        tracked_points.push_back(map_points[index]);
    }

    mCurrentFrame.set_tracked_features(std::move(tracked_keys), std::move(tracked_points), std::move(tracked_display_keys));
    return mCurrentFrame.N >= 0.6 * flow_reference_matches_ ? mCurrentFrame.N : 0;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    UpdateLocalMap();
    SearchLocalPoints();

    // TOO check outliers before PO
    int aux1 = 0, aux2 = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if (mCurrentFrame.mvbOutlier[i])
            {
                aux2++;
            }
        }
    }

    if (!mpAtlas->isImuInitialized())
    {
        Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);
    }
    else
    {
        if (mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU)
        {
            spdlog::debug("TLM: PoseOptimization ");
            Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);
        }
        else
        {
            // A flow recovery can reoptimize this frame after the last-frame prior was consumed.
            delete mCurrentFrame.mpcpi;
            mCurrentFrame.mpcpi = nullptr;
            if (!mbMapUpdated && mLastFrame.mpcpi)
            {
                spdlog::trace("TLM: PoseInertialOptimizationLastFrame ");
                Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame, mpCamera, mpCamera2);
            }
            else
            {
                spdlog::trace("TLM: PoseInertialOptimizationLastKeyFrame ");
                Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame, mpCamera, mpCamera2);
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if (mCurrentFrame.mvbOutlier[i])
            {
                aux2++;
            }
        }
    }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    {
                        mnMatchesInliers++;
                    }
                }
                else
                {
                    mnMatchesInliers++;
                }
            }
            else if (mSensor == System::STEREO)
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
    if (mCurrentFrame.flow_only && !mCurrentFrame.reliable_flow_inliers(mImGray.size()))
    {
        return false;
    }
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
    {
        return false;
    }

    if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
    {
        return true;
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        if ((mnMatchesInliers < 15 && mpAtlas->isImuInitialized()) || (mnMatchesInliers < 50 && !mpAtlas->isImuInitialized()))
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if (mnMatchesInliers < 15)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    else
    {
        if (mnMatchesInliers < 30)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
        {
            return true;
        }
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    if (mbOnlyTracking)
    {
        return false;
    }

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    {
        /*if(mSensor == System::MONOCULAR)
        {
            spdlog::debug("Keyframe creation deferred because local mapping is stopped");
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
    {
        nMinObs = 2;
    }
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    // Flow tracks a bounded left-camera subset, not all ORB observations in both cameras.
    if (mCurrentFrame.flow_only)
    {
        nRefMatches = std::min(nRefMatches, flow_reference_matches_);
    }

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int  nNonTrackedClose  = 0;
    int  nTrackedClose     = 0;

    if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for (int i = 0; i < N; i++)
        {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
            {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                {
                    nTrackedClose++;
                }
                else
                {
                    nNonTrackedClose++;
                }
            }
        }
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Thresholds
    float thRefRatio   = 0.75f;
    if (nKFs < 2)
    {
        thRefRatio = 0.4f;
    }

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if (mSensor == System::MONOCULAR)
    {
        thRefRatio = 0.9f;
    }

    if (mpCamera2)
    {
        thRefRatio = 0.75f;
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        if (mnMatchesInliers > 350) // Points tracked from the local map
        {
            thRefRatio = 0.75f;
        }
        else
        {
            thRefRatio = 0.90f;
        }
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO && mSensor != System::IMU_RGBD &&
                     (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) && mnMatchesInliers > 15);

    //  Temporal condition for Inertial cases
    bool       c3 = false;
    if (mpLastKeyFrame)
    {
        if (mSensor == System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
            {
                c3 = true;
            }
        }
        else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
            {
                c3 = true;
            }
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR))
    {
        c4 = true;
    }
    else
    {
        c4 = false;
    }

    if (((c1a || c1b || c1c) && c2) || c3 || c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }
}

void Tracking::CreateNewKeyFrame()
{
    if (mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
    {
        return;
    }
    if (mCurrentFrame.flow_only)
    {
        const Frame flow_frame(mCurrentFrame);
        extract_current_features();
        descriptor_matcher matcher(0.9, true);
        matcher.search_by_projection(mCurrentFrame, mLastFrame, mSensor == System::STEREO ? 7 : 15, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR, mpCamera);
        SearchLocalPoints();
        const int inliers = Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);
        if (inliers < 50)
        {
            spdlog::warn("Frame {}: postponing keyframe; only {} descriptor inliers after feature refresh", mCurrentFrame.mnId, inliers);
            mCurrentFrame = flow_frame;
            return;
        }
        mnMatchesInliers = inliers;
        mVelocity        = mCurrentFrame.GetPose() * mLastFrame.GetPose().inverse();
        if (!NeedNewKeyFrame())
        {
            return;
        }
    }

    if (!mpLocalMapper->SetNotStop(true))
    {
        return;
    }

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB, mpCamera, mpCamera2);

    if (mpAtlas->isImuInitialized())
    {
        pKF->bImu = true;
    }

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF               = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (mpLastKeyFrame)
    {
        pKF->mPrevKF            = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
    {
        spdlog::debug("No last KF in KF creation!!");
    }

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
    }

    if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if include imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            maxPoint = 100;
        }

        vector<pair<float, int>> vDepthIdx;
        int                      N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int       i          = vDepthIdx[j].second;

                bool      bCreateNew = false;

                MapPoint *pMP        = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                {
                    bCreateNew = true;
                }
                else if (pMP->Observations() < 1)
                {
                    bCreateNew                    = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if (mCurrentFrame.Nleft == -1)
                    {
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else
                    {
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF, i);

                    // Check if it is a stereo observation in order to not
                    // duplicate mappoints
                    if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0)
                    {
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                        pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint)
                {
                    break;
                }
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<MapPoint *>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView   = false;
                pMP->mbTrackInViewR  = false;
            }
        }
    }

    if (mCurrentFrame.flow_only)
    {
        return;
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;

        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
        {
            continue;
        }
        if (pMP->isBad())
        {
            continue;
        }
        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5, mpCamera, mpCamera2))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        if (pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    if (nToMatch > 0)
    {
        descriptor_matcher matcher(0.8);
        int                th = 1;
        if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
        {
            th = 3;
        }
        if (mpAtlas->isImuInitialized())
        {
            if (mpAtlas->GetCurrentMap()->GetIniertialBA2())
            {
                th = 2;
            }
            else
            {
                th = 6;
            }
        }
        else if (!mpAtlas->isImuInitialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th = 10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
        {
            th = 5;
        }

        if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
        {
            th = 15;                                   // 15
        }

        matcher.search_by_projection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF)
    {
        KeyFrame                *pKF   = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {

            MapPoint *pMP = *itMP;
            if (!pMP)
            {
                continue;
            }
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
            {
                continue;
            }
            if (!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    if (!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2))
    {
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                    for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    {
                        keyframeCounter[it->first]++;
                    }
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if (mLastFrame.mvpMapPoints[i])
            {
                MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                if (!pMP)
                {
                    continue;
                }
                if (!pMP->isBad())
                {
                    const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                    for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    {
                        keyframeCounter[it->first]++;
                    }
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i] = NULL;
                }
            }
        }
    }

    int       max    = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
        {
            continue;
        }

        if (it->second > max)
        {
            max    = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80) // 80
        {
            break;
        }

        KeyFrame                *pKF     = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame *> spChilds = pKF->GetChilds();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mvpLocalKeyFrames.size() < 80)
    {
        KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd           = 20;
        for (int i = 0; i < Nd; i++)
        {
            if (!tempKeyFrame)
            {
                break;
            }
            if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                tempKeyFrame                           = tempKeyFrame->mPrevKF;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF               = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    extract_current_features();
    spdlog::debug("Starting relocalization");
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if (vpCandidateKFs.empty())
    {
        spdlog::debug("There are not candidates");
        return false;
    }

    const int                  nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    descriptor_matcher         matcher(0.75, true);

    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
        {
            vbDiscarded[i] = true;
        }
        else
        {
            int nmatches = matcher.search_by_vocabulary(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
            }
        }
    }

    bool               bMatch = false;
    descriptor_matcher matcher2(0.9, true);

    for (int i = 0; i < nKFs && !bMatch; ++i)
    {
        if (vbDiscarded[i])
        {
            continue;
        }

        vector<cv::Point3f> world_points;
        vector<cv::Point2f> image_points;
        vector<float>       max_errors;
        world_points.reserve(vvpMapPointMatches[i].size());
        image_points.reserve(vvpMapPointMatches[i].size());
        max_errors.reserve(vvpMapPointMatches[i].size());
        for (size_t j = 0; j < vvpMapPointMatches[i].size(); ++j)
        {
            MapPoint *map_point = vvpMapPointMatches[i][j];
            if (!map_point || map_point->isBad() || j >= mCurrentFrame.mvKeysUn.size())
            {
                continue;
            }
            const cv::KeyPoint   &keypoint = mCurrentFrame.mvKeysUn[j];
            const Eigen::Vector3f position = map_point->GetWorldPos();
            world_points.emplace_back(position.x(), position.y(), position.z());
            image_points.push_back(keypoint.pt);
            max_errors.push_back(mCurrentFrame.mvLevelSigma2[keypoint.octave] * 5.991f);
        }

        vector<bool>    pnp_inliers;
        Eigen::Matrix4f Tcw_matrix;
        const int       min_inliers = std::max(10, static_cast<int>(world_points.size() * 0.5));
        if (!GeometricTools::solve_pnp_ransac(world_points, image_points, max_errors, mpCamera, min_inliers, Tcw_matrix, pnp_inliers))
        {
            continue;
        }

        mCurrentFrame.SetPose(Sophus::SE3f(Tcw_matrix));
        set<MapPoint *> sFound;
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        size_t correspondence_index = 0;
        for (size_t j = 0; j < vvpMapPointMatches[i].size(); ++j)
        {
            MapPoint *map_point = vvpMapPointMatches[i][j];
            if (!map_point || map_point->isBad() || j >= mCurrentFrame.mvKeysUn.size())
            {
                continue;
            }
            if (pnp_inliers[correspondence_index])
            {
                mCurrentFrame.mvpMapPoints[j] = map_point;
                sFound.insert(map_point);
            }
            ++correspondence_index;
        }

        int nGood = Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);

        if (nGood < 10)
        {
            continue;
        }

        for (int io = 0; io < mCurrentFrame.N; io++)
        {
            if (mCurrentFrame.mvbOutlier[io])
            {
                mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);
            }
        }

        // If few inliers, search by projection in a coarse window and optimize again
        if (nGood < 50)
        {
            int nadditional = matcher2.search_by_projection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100, mpCamera);

            if (nadditional + nGood >= 50)
            {
                nGood = Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);

                // If many inliers but still not enough, search by projection again in a narrower window
                // the camera has been already optimized with many points
                if (nGood > 30 && nGood < 50)
                {
                    sFound.clear();
                    for (int ip = 0; ip < mCurrentFrame.N; ip++)
                    {
                        if (mCurrentFrame.mvpMapPoints[ip])
                        {
                            sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                        }
                    }
                    nadditional = matcher2.search_by_projection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64, mpCamera);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame, mpCamera, mpCamera2);

                        for (int io = 0; io < mCurrentFrame.N; io++)
                        {
                            if (mCurrentFrame.mvbOutlier[io])
                            {
                                mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }
            }
        }

        // If the pose is supported by enough inliers stop and continue
        if (nGood >= 50)
        {
            bMatch = true;
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        spdlog::info("Relocalized");
        return true;
    }
}

void Tracking::Reset(bool bLocMap)
{
    spdlog::debug("System Reseting");
    if (mOnResetStart)
    {
        mOnResetStart();
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        spdlog::debug("Reseting Local Mapper...");
        mpLocalMapper->RequestReset();
        spdlog::debug("done");
    }

    // Reset Loop Closing
    spdlog::debug("Reseting Loop Closing...");
    mpLoopClosing->RequestReset();
    spdlog::debug("done");

    // Clear BoW Database
    spdlog::debug("Reseting Database...");
    mpKeyFrameDB->clear();
    spdlog::debug("done");

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
    {
        mpAtlas->SetInertialSensor();
    }
    mnInitialFrameId      = 0;

    KeyFrame::nNextId     = 0;
    Frame::nNextId        = 0;
    mState                = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit             = false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame      = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame         = Frame();
    _previous_gray.release();
    mpReferenceKF  = static_cast<KeyFrame *>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    mvIniMatches.clear();

    if (mOnResetComplete)
    {
        mOnResetComplete();
    }
    spdlog::debug("   End reseting! ");
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    spdlog::debug("Active map Reseting");
    if (mOnResetStart)
    {
        mOnResetStart();
    }
    Map *pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        spdlog::debug("Reseting Local Mapper...");
        mpLocalMapper->RequestResetActiveMap(pMap);
        spdlog::trace("done");
    }

    // Reset Loop Closing
    spdlog::debug("Reseting Loop Closing...");
    mpLoopClosing->RequestResetActiveMap(pMap);
    spdlog::debug("done");

    // Clear BoW Database
    spdlog::debug("Reseting Database");
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    spdlog::debug("done");

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();

    // KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    // Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId     = Frame::nNextId;
    mState                = NO_IMAGES_YET; // NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool>   lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    spdlog::debug("First frame ID: {}", mnFirstFrameId);
    for (Map *pMap : mpAtlas->GetAllMaps())
    {
        if (pMap->GetAllKeyFrames().size() > 0)
        {
            if (index > pMap->GetLowerKFID())
            {
                index = pMap->GetLowerKFID();
            }
        }
    }

    int num_lost = 0;
    spdlog::debug("Initial frame ID: {}", mnInitialFrameId);

    for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if (index < mnInitialFrameId)
        {
            lbLost.push_back(*ilbL);
        }
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    spdlog::debug("{} frames set to lost", num_lost);

    mlbLost            = lbLost;

    mnInitialFrameId   = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame      = Frame();
    mLastFrame         = Frame();
    _previous_gray.release();
    mpReferenceKF  = static_cast<KeyFrame *>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if (mOnResetComplete)
    {
        mOnResetComplete();
    }
    spdlog::debug("   End reseting! ");
}

vector<MapPoint *> Tracking::GetLocalMapMPS() { return mvpLocalMapPoints; }

void               Tracking::InformOnlyTracking(const bool &flag) { mbOnlyTracking = flag; }

void               Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
{
    Map                                  *pMap = pCurrentKeyFrame->GetMap();
    list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator                  lbL  = mlbLost.begin();
    for (auto lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lbL++)
    {
        if (*lbL)
        {
            continue;
        }

        KeyFrame *pKF = *lRit;

        while (pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if (pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias      = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while (!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }

    if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(), mLastFrame.mpLastKeyFrame->GetImuPosition(), mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float                 t12  = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float                 t12  = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                         twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                         Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
    _previous_gray.release();
}

int  Tracking::GetNumberDataset() { return mnNumDataset; }

int  Tracking::GetMatchesInliers() { return mnMatchesInliers; }

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    // mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map *pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if (!strNameFile_kf.empty())
    {
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
    }
}

float Tracking::GetImageScale() { return mImageScale; }

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        spdlog::debug("Tracking stopped");
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped       = false;
    mbStopRequested = false;
}
#endif

} // namespace ORB_SLAM3
