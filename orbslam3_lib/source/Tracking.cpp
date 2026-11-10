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

#include "G2oTypes.h"
#include "KannalaBrandt8.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "converter.h"
#include "descriptor_matcher.h"
#include "geometric_tools.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <mutex>

#include <opencv2/video/tracking.hpp>

#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>

using namespace std;

namespace ORB_SLAM3
{

Tracking::Tracking(System *pSys, place_vocabulary *pVoc, atlas *pAtlas, KeyFrameDatabase *pKFDB, const int sensor, Settings *settings, const camera_vector &cameras, const string &_nameSeq)
    : mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false), mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), place_vocabulary_ptr(pVoc), mpKeyFrameDB(pKFDB),
      mbReadyToInitializate(false), mpSystem(pSys), bStepByStep(false), mpAtlas(pAtlas), mpLastKeyFrame(nullptr), mnLastRelocFrameId(0), time_recently_lost(5.0), mnFirstFrameId(0),
      mnInitialFrameId(0), mbCreatedMap(false), cameras(cameras)
{
    load_parameters(settings);

    initID         = 0;
    lastID         = 0;
    mbInitWith3KFs = false;
    mnNumDataset   = 0;

    for (const auto &pCam : cameras)
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
void Tracking::load_parameters(Settings *settings)
{

    if (settings->needToUndistort())
    {
        mDistCoef = settings->camera1DistortionCoef();
    }
    else
    {
        mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    }

    // TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK                 = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0, 0) = cameras[0]->getParameter(0);
    mK.at<float>(1, 1) = cameras[0]->getParameter(1);
    mK.at<float>(0, 2) = cameras[0]->getParameter(2);
    mK.at<float>(1, 2) = cameras[0]->getParameter(3);

    mK_.setIdentity();
    mK_(0, 0) = cameras[0]->getParameter(0);
    mK_(1, 1) = cameras[0]->getParameter(1);
    mK_(0, 2) = cameras[0]->getParameter(2);
    mK_(1, 2) = cameras[0]->getParameter(3);

    if ((mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && settings->cameraType() == Settings::KannalaBrandt)
    {

        mTlr = settings->Tlr();
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

    mMinFrames         = 0;
    mMaxFrames         = settings->fps();
    mnFramesToResetIMU = mMaxFrames;
    mbRGB              = settings->rgb();

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
    if (place_vocabulary_ptr && (place_vocabulary_ptr->descriptor_type() != left_keypoint_extractor->get_descriptor_type() ||
                                 place_vocabulary_ptr->descriptor_size() != left_keypoint_extractor->get_descriptor_size()))
    {
        throw std::runtime_error("The place-recognition vocabulary descriptor format does not match keypoint.descriptor.type");
    }

    if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
    {
        right_keypoint_extractor = std::make_unique<keypoint_extractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);
    }

    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        initial_keypoint_extractor = std::make_unique<keypoint_extractor>(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);
    }

    // IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost   = settings->insertKFsWhenLost();
    mImuFreq         = settings->imuFrequency();
    mImuPer          = 0.001; // TODO: ESTO ESTA BIEN?
    float Ng         = settings->noiseGyro();
    float Na         = settings->noiseAcc();
    float Ngw        = settings->gyroWalk();
    float Naw        = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib     = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) { mpLocalMapper = pLocalMapper; }

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) { mpLoopClosing = pLoopClosing; }

void Tracking::SetResetHooks(std::function<void()> on_reset_start, std::function<void()> on_reset_complete)
{
    mOnResetStart    = std::move(on_reset_start);
    mOnResetComplete = std::move(on_reset_complete);
}

void Tracking::SetStepByStep(bool bSet) { bStepByStep = bSet; }

bool Tracking::GetStepByStep() { return bStepByStep; }

void Tracking::set_frame_colors(const cv::Mat &left, const cv::Mat &right)
{
    const auto to_bgr = [this](const cv::Mat &image)
    {
        cv::Mat result;
        if (image.empty())
        {
            return result;
        }
        if (image.channels() == 1)
        {
            cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, result, mbRGB ? cv::COLOR_RGBA2BGR : cv::COLOR_BGRA2BGR);
        }
        else if (mbRGB)
        {
            cv::cvtColor(image, result, cv::COLOR_RGB2BGR);
        }
        else
        {
            result = image.clone();
        }
        return result;
    };
    mImColor                        = mNextFrameColor.empty() ? to_bgr(left) : std::move(mNextFrameColor);
    mCurrentFrame.color_image       = mImColor;
    mCurrentFrame.color_image_right = next_frame_color_right.empty() ? to_bgr(right) : std::move(next_frame_color_right);
    mCurrentFrame.color_is_rgb      = false;
    mNextFrameColor.release();
    next_frame_color_right.release();
}

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
    if (can_defer_features(timestamp))
    {
        mCurrentFrame = frame(mLastFrame, timestamp);
    }
    else
    {
        create_frame(timestamp, *left_keypoint_extractor);
    }

    set_frame_colors(imRectLeft, imRectRight);
    mCurrentFrame.name_file = filename;
    mCurrentFrame.dataset   = mnNumDataset;

    mCurrentFrame.optical_flow_pyramid.swap(next_optical_flow_pyramid);
    if (!mCurrentFrame.flow_only)
    {
        last_feature_frame_id_ = mCurrentFrame.id;
        ++feature_frames_;
    }
    Track();
    if (mCurrentFrame.flow_only && mState == OK)
    {
        ++flow_frames_;
    }

    if (_use_optical_flow && mLastFrame.id == mCurrentFrame.id && !mImGray.empty())
    {
        _previous_gray          = mImGray.clone();
        previous_right_gray_    = mImRight.clone();
        _previous_gray_frame_id = mCurrentFrame.id;
    }
    else
    {
        _previous_gray.release();
        previous_right_gray_.release();
    }

    return mCurrentFrame.get_pose();
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

    create_frame(timestamp, *left_keypoint_extractor);
    mCurrentFrame.compute_stereo_from_rgbd(imDepth);

    set_frame_colors(imRGB);
    mCurrentFrame.name_file = filename;
    mCurrentFrame.dataset   = mnNumDataset;

    Track();

    return mCurrentFrame.get_pose();
}

Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;

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
        mCurrentFrame = frame(mLastFrame, timestamp);
    }
    else
    {
        const bool initializing = mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (mSensor == System::MONOCULAR && (lastID - initID) < mMaxFrames);
        create_frame(timestamp, initializing ? *initial_keypoint_extractor : *left_keypoint_extractor);
    }

    if (mState == NO_IMAGES_YET)
    {
        t0 = timestamp;
    }

    set_frame_colors(im);
    mCurrentFrame.name_file = filename;
    mCurrentFrame.dataset   = mnNumDataset;

    lastID = mCurrentFrame.id;
    mCurrentFrame.optical_flow_pyramid.swap(next_optical_flow_pyramid);
    if (!mCurrentFrame.flow_only)
    {
        last_feature_frame_id_ = mCurrentFrame.id;
        ++feature_frames_;
    }
    Track();
    if (mCurrentFrame.flow_only && mState == OK)
    {
        ++flow_frames_;
    }

    if (_use_optical_flow && mLastFrame.id == mCurrentFrame.id && !mImGray.empty())
    {
        _previous_gray          = mImGray.clone();
        _previous_gray_frame_id = mCurrentFrame.id;
    }
    else
    {
        _previous_gray.release();
    }

    return mCurrentFrame.get_pose();
}

void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if (!mCurrentFrame.previous_frame)
    {
        spdlog::debug("non prev frame");
        mCurrentFrame.set_integrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if (mlQueueImuData.size() == 0)
    {
        spdlog::debug("Not IMU data in mlQueueImuData!!");
        mCurrentFrame.set_integrated();
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
                if (m->t < mCurrentFrame.previous_frame->timestamp - mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if (m->t < mCurrentFrame.timestamp - mImuPer)
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

    IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.imu_bias, mCurrentFrame.imu_calibration);

    for (int i = 0; i < n; i++)
    {
        float           tstep;
        Eigen::Vector3f acc, angVel;
        if ((i == 0) && (i < (n - 1)))
        {
            float tab  = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t - mCurrentFrame.previous_frame->timestamp;
            acc        = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
            angVel     = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
            tstep      = mvImuFromLastFrame[i + 1].t - mCurrentFrame.previous_frame->timestamp;
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
            float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.timestamp;
            acc        = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
            angVel     = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
            tstep      = mCurrentFrame.timestamp - mvImuFromLastFrame[i].t;
        }
        else if ((i == 0) && (i == (n - 1)))
        {
            acc    = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep  = mCurrentFrame.timestamp - mCurrentFrame.previous_frame->timestamp;
        }

        if (!mpImuPreintegratedFromLastKF)
        {
            spdlog::error("IMU preintegration from the last keyframe is missing");
        }
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
    }

    mCurrentFrame.frame_imu_preintegrated = pImuPreintegratedFromLastFrame;
    mCurrentFrame.imu_preintegrated       = mpImuPreintegratedFromLastKF;
    mCurrentFrame.last_key_frame          = mpLastKeyFrame;

    mCurrentFrame.set_integrated();
}

bool Tracking::PredictStateIMU()
{
    if (!mCurrentFrame.previous_frame)
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
        const float           t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.set_imu_pose_velocity(Rwb2, twb2, Vwb2);

        mCurrentFrame.imu_bias           = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.predicted_imu_bias = mCurrentFrame.imu_bias;
        return true;
    }
    else if (!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.get_imu_position();
        const Eigen::Matrix3f Rwb1 = mLastFrame.get_imu_rotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.get_velocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float           t12 = mCurrentFrame.frame_imu_preintegrated->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.frame_imu_preintegrated->GetDeltaRotation(mLastFrame.imu_bias));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.frame_imu_preintegrated->GetDeltaPosition(mLastFrame.imu_bias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.frame_imu_preintegrated->GetDeltaVelocity(mLastFrame.imu_bias);

        mCurrentFrame.set_imu_pose_velocity(Rwb2, twb2, Vwb2);

        mCurrentFrame.imu_bias           = mLastFrame.imu_bias;
        mCurrentFrame.predicted_imu_bias = mCurrentFrame.imu_bias;
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

    map *pCurrentMap = mpAtlas->get_current_map();
    if (!pCurrentMap)
    {
        spdlog::error("There is no active map in the Atlas");
    }

    if (mState != NO_IMAGES_YET)
    {
        if (mLastFrame.timestamp > mCurrentFrame.timestamp)
        {
            spdlog::error("Frame with a timestamp older than the previous frame detected");
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if (mCurrentFrame.timestamp > mLastFrame.timestamp + 1.0)
        {
            if (mpAtlas->is_inertial())
            {

                if (mpAtlas->is_imu_initialized())
                {
                    spdlog::warn("Timestamp jump detected; setting state to LOST and resetting IMU integration");
                    if (!pCurrentMap->get_inertial_ba2())
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
        mCurrentFrame.set_new_bias(mpLastKeyFrame->GetImuBias());
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
    unique_lock<mutex> lock(pCurrentMap->map_update_mutex);

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->get_map_change_index();
    int nMapChangeIndex    = pCurrentMap->get_last_map_change();
    if (nCurMapChangeIndex > nMapChangeIndex)
    {
        pCurrentMap->set_last_map_change(nCurMapChangeIndex);
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
            mLastFrame = frame(mCurrentFrame);
            return;
        }

        if (mpAtlas->get_all_maps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.id;
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

                if ((!mbVelocity && !pCurrentMap->is_imu_initialized()) || mCurrentFrame.id < mnLastRelocFrameId + 2)
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
                    if (mCurrentFrame.id <= (mnLastRelocFrameId + mnFramesToResetIMU) && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if (pCurrentMap->key_frame_count() > 10)
                    {
                        mState         = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.timestamp;
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
                        if (pCurrentMap->is_imu_initialized())
                        {
                            PredictStateIMU();
                        }
                        else
                        {
                            bOK = false;
                        }

                        if (mCurrentFrame.timestamp - mTimeStampLost > time_recently_lost)
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
                        if (mCurrentFrame.timestamp - mTimeStampLost > 3.0f && !bOK)
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

                    if (pCurrentMap->key_frame_count() < 10)
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
                        vpMPsMM = mCurrentFrame.map_points;
                        vbOutMM = mCurrentFrame.outliers;
                        TcwMM   = mCurrentFrame.get_pose();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.set_pose(TcwMM);
                        mCurrentFrame.map_points = vpMPsMM;
                        mCurrentFrame.outliers   = vbOutMM;

                        if (mbVO)
                        {
                            for (int i = 0; i < mCurrentFrame.feature_count; i++)
                            {
                                if (mCurrentFrame.map_points[i] && !mCurrentFrame.outliers[i])
                                {
                                    mCurrentFrame.map_points[i]->IncreaseFound();
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

        if (!mCurrentFrame.reference_key_frame)
        {
            mCurrentFrame.reference_key_frame = mpReferenceKF;
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
            spdlog::debug("Frame {}: extracting features for local-map recovery", mCurrentFrame.id);
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
                if (!pCurrentMap->is_imu_initialized() || !pCurrentMap->get_inertial_ba2())
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

            mTimeStampLost = mCurrentFrame.timestamp;
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if ((mCurrentFrame.id < (mnLastRelocFrameId + mnFramesToResetIMU)) && (mCurrentFrame.id > static_cast<unsigned int>(mnFramesToResetIMU)) &&
            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->is_imu_initialized())
        {
            // TODO check this situation
            spdlog::debug("Saving pointer to frame. imu needs reset...");
            frame *pF          = new frame(mCurrentFrame);
            pF->previous_frame = new frame(mLastFrame);

            // Load preintegration
            pF->frame_imu_preintegrated = new IMU::Preintegrated(mCurrentFrame.frame_imu_preintegrated);
        }

        if (pCurrentMap->is_imu_initialized())
        {
            if (bOK)
            {
                if (mCurrentFrame.id == (mnLastRelocFrameId + mnFramesToResetIMU))
                {
                    spdlog::debug("Resetting frame");
                    ResetFrameIMU();
                }
                else if (mCurrentFrame.id > (mnLastRelocFrameId + 30))
                {
                    mLastBias = mCurrentFrame.imu_bias;
                }
            }
        }

        if (bOK || mState == RECENTLY_LOST)
        {
            // Update motion model
            if (mLastFrame.is_set() && mCurrentFrame.is_set())
            {
                Sophus::SE3f LastTwc = mLastFrame.get_pose().inverse();
                mVelocity            = mCurrentFrame.get_pose() * LastTwc;
                mbVelocity           = true;
            }
            else
            {
                mbVelocity = false;
            }

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.feature_count; i++)
            {
                MapPoint *pMP = mCurrentFrame.map_points[i];
                if (pMP)
                {
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.outliers[i]   = false;
                        mCurrentFrame.map_points[i] = static_cast<MapPoint *>(NULL);
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
            for (int i = 0; i < mCurrentFrame.feature_count; i++)
            {
                if (mCurrentFrame.map_points[i] && mCurrentFrame.outliers[i])
                {
                    mCurrentFrame.map_points[i] = static_cast<MapPoint *>(NULL);
                }
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (pCurrentMap->key_frame_count() <= 10)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                if (!pCurrentMap->is_imu_initialized())
                {
                    spdlog::warn("Track lost before IMU initialisation; resetting");
                    mpSystem->ResetActiveMap();
                    return;
                }
            }

            CreateMapInAtlas();

            return;
        }

        if (!mCurrentFrame.reference_key_frame)
        {
            mCurrentFrame.reference_key_frame = mpReferenceKF;
        }

        mLastFrame = frame(mCurrentFrame);
    }

    if (mState == OK || mState == RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (mCurrentFrame.is_set())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.get_pose() * mCurrentFrame.reference_key_frame->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.reference_key_frame);
            mlFrameTimes.push_back(mCurrentFrame.timestamp);
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
    if (mCurrentFrame.feature_count > 500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.imu_preintegrated || !mLastFrame.imu_preintegrated)
            {
                spdlog::debug("No IMU measurements");
                return;
            }

            if (!mFastInit && (mCurrentFrame.frame_imu_preintegrated->avgA - mLastFrame.frame_imu_preintegrated->avgA).norm() < 0.5)
            {
                spdlog::debug("Insufficient acceleration for initialization");
                return;
            }

            if (mpImuPreintegratedFromLastKF)
            {
                delete mpImuPreintegratedFromLastKF;
            }

            mpImuPreintegratedFromLastKF    = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
            mCurrentFrame.imu_preintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.imu_calibration.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.imu_calibration.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.set_imu_pose_velocity(Rwb0, twb0, Vwb0);
        }
        else
        {
            mCurrentFrame.set_pose(Sophus::SE3f());
        }

        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->get_current_map(), mpKeyFrameDB, cameras);

        // Insert KeyFrame in the map
        mpAtlas->add_key_frame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if (!is_fisheye_stereo(cameras))
        {
            for (int i = 0; i < mCurrentFrame.feature_count; i++)
            {
                float z = mCurrentFrame.depths[i];
                if (z > 0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.unproject_stereo(i, x3D);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->get_current_map());
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->add_map_point(pNewMP);

                    mCurrentFrame.map_points[i] = pNewMP;
                }
            }
        }
        else
        {
            for (int i = 0; i < static_cast<int>(mCurrentFrame.keypoints.size()); i++)
            {
                int rightIndex = mCurrentFrame.left_to_right_matches[i];
                if (rightIndex != -1)
                {
                    Eigen::Vector3f x3D = mCurrentFrame.stereo_3d_points[i];

                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->get_current_map());

                    pNewMP->AddObservation(pKFini, i);
                    pNewMP->AddObservation(pKFini, rightIndex + static_cast<int>(mCurrentFrame.keypoints.size()));

                    pKFini->AddMapPoint(pNewMP, i);
                    pKFini->AddMapPoint(pNewMP, rightIndex + static_cast<int>(mCurrentFrame.keypoints.size()));

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->add_map_point(pNewMP);

                    mCurrentFrame.map_points[i]                                                             = pNewMP;
                    mCurrentFrame.map_points[rightIndex + static_cast<int>(mCurrentFrame.keypoints.size())] = pNewMP;
                }
            }
        }

        spdlog::info("New Map created with {} points", mpAtlas->map_points_in_map());

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame       = frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.id;
        mpLastKeyFrame   = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints                 = mpAtlas->get_all_map_points();
        mpReferenceKF                     = pKFini;
        mCurrentFrame.reference_key_frame = pKFini;

        mpAtlas->set_reference_map_points(mvpLocalMapPoints);

        mpAtlas->get_current_map()->key_frame_origins.push_back(pKFini);

        mState = OK;
    }
}

void Tracking::MonocularInitialization()
{

    if (!mbReadyToInitializate)
    {
        // Set Reference Frame
        if (mCurrentFrame.keypoints.size() > 100)
        {

            mInitialFrame = frame(mCurrentFrame);
            mLastFrame    = frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.keypoints.size());
            for (size_t i = 0; i < mCurrentFrame.keypoints.size(); i++)
            {
                mvbPrevMatched[i] = mCurrentFrame.keypoints[i].pt_undistorted;
            }

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if (mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF    = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                mCurrentFrame.imu_preintegrated = mpImuPreintegratedFromLastKF;
            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.keypoints.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.timestamp - mInitialFrame.timestamp > 1.0)))
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

        const auto initial_keys = to_cv_keypoints(mInitialFrame.keypoints, true);
        const auto current_keys = to_cv_keypoints(mCurrentFrame.keypoints, true);
        if (cameras[0]->ReconstructWithTwoViews(initial_keys, current_keys, mvIniMatches, Tcw, mvIniP3D, vbTriangulated))
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
            mInitialFrame.set_pose(Sophus::SE3f());
            mCurrentFrame.set_pose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->get_current_map(), mpKeyFrameDB, cameras);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->get_current_map(), mpKeyFrameDB, cameras);

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);
    }

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->add_key_frame(pKFini);
    mpAtlas->add_key_frame(pKFcur);

    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
        {
            continue;
        }

        // Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->get_current_map());

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        mCurrentFrame.map_points[mvIniMatches[i]] = pMP;
        mCurrentFrame.outliers[mvIniMatches[i]]   = false;

        // Add to Map
        mpAtlas->add_map_point(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint *> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    spdlog::info("New Map created with {} points", mpAtlas->map_points_in_map());
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->get_current_map(), 20);

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
        pKFcur->mPrevKF            = pKFini;
        pKFini->mNextKF            = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

    mCurrentFrame.set_pose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.id;
    mpLastKeyFrame   = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints                 = mpAtlas->get_all_map_points();
    mpReferenceKF                     = pKFcur;
    mCurrentFrame.reference_key_frame = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame *> vKFs = mpAtlas->get_all_key_frames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity          = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux  = (mCurrentFrame.timestamp - mLastFrame.timestamp) / (mCurrentFrame.timestamp - mInitialFrame.timestamp);
    phi        *= aux;

    mLastFrame = frame(mCurrentFrame);

    mpAtlas->set_reference_map_points(mvpLocalMapPoints);

    mpAtlas->get_current_map()->key_frame_origins.push_back(pKFini);

    mState = OK;

    initID = pKFcur->mnId;
}

void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.id;
    mpAtlas->create_new_map();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
    {
        mpAtlas->set_inertial_sensor();
    }
    mbSetInit = false;

    mnInitialFrameId = mCurrentFrame.id + 1;
    mState           = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
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

    mLastFrame    = frame();
    mCurrentFrame = frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.feature_count; i++)
    {
        MapPoint *pMP = mLastFrame.map_points[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.map_points[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    extract_current_features();
    // Compute Bag of Words vector
    mCurrentFrame.compute_bow();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    descriptor_matcher matcher(0.7, true);
    vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.search_by_vocabulary(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        spdlog::debug("Reference keyframe tracking found fewer than 15 matches");
        return false;
    }

    mCurrentFrame.map_points = vpMapPointMatches;
    mCurrentFrame.set_pose(mLastFrame.get_pose());

    Optimizer::PoseOptimization(&mCurrentFrame, cameras);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.feature_count; i++)
    {
        if (mCurrentFrame.map_points[i])
        {
            if (mCurrentFrame.outliers[i])
            {
                MapPoint *pMP = mCurrentFrame.map_points[i];

                mCurrentFrame.map_points[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.outliers[i]   = false;
                if (mCurrentFrame.is_fisheye_stereo && std::cmp_less(i, mCurrentFrame.keypoints.size()))
                {
                    pMP->mbTrackInView = false;
                }
                else
                {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView   = false;
                pMP->mnLastFrameSeen = mCurrentFrame.id;
                nmatches--;
            }
            else if (mCurrentFrame.map_points[i]->Observations() > 0)
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
    KeyFrame    *pRef = mLastFrame.reference_key_frame;
    Sophus::SE3f Tlr  = mlRelativeFramePoses.back();
    mLastFrame.set_pose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.id || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking)
    {
        return;
    }

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int>> vDepthIdx;
    const int                Nfeat = mLastFrame.is_fisheye_stereo ? static_cast<int>(mLastFrame.keypoints.size()) : mLastFrame.feature_count;
    vDepthIdx.reserve(Nfeat);
    for (int i = 0; i < Nfeat; i++)
    {
        float z = mLastFrame.depths[i];
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
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.map_points[i];

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

            if (!mLastFrame.is_fisheye_stereo)
            {
                mLastFrame.unproject_stereo(i, x3D);
            }
            else
            {
                x3D = mLastFrame.unproject_stereo_fisheye(i);
            }

            MapPoint *pNewMP         = new MapPoint(x3D, mpAtlas->get_current_map(), &mLastFrame, i);
            mLastFrame.map_points[i] = pNewMP;

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

    const bool imu_prediction = mpAtlas->is_imu_initialized() && (mCurrentFrame.id > mnLastRelocFrameId + mnFramesToResetIMU);

    if (imu_prediction)
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
    }
    else
    {
        mCurrentFrame.set_pose(mVelocity * mLastFrame.get_pose());
    }

    if (mCurrentFrame.flow_only)
    {
        const auto predicted_pose = mCurrentFrame.get_pose();
        const int  flow_matches   = track_with_optical_flow();
        bool       reliable_flow  = false;
        const bool stereo         = mSensor == System::STEREO || mSensor == System::IMU_STEREO;

        if (flow_matches >= 50)
        {
            Optimizer::PoseOptimization(&mCurrentFrame, cameras);

            const int minimum_inliers = stereo ? std::clamp(flow_reference_matches_ / 2, 100, 300) : 50;
            reliable_flow             = mCurrentFrame.reliable_flow_inliers(mImGray.size(), minimum_inliers);
        }

        if (reliable_flow)
        {
            spdlog::debug("Frame {}: flow-only tracking, {} observations", mCurrentFrame.id, flow_matches);
            return true;
        }

        spdlog::debug("Frame {}: optical flow weak; extracting features for descriptor recovery", mCurrentFrame.id);
        ++flow_recoveries_;
        extract_current_features();
        mCurrentFrame.set_pose(predicted_pose);
    }

    fill(mCurrentFrame.map_points.begin(), mCurrentFrame.map_points.end(), static_cast<MapPoint *>(NULL));
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
    int nmatches = matcher.search_by_projection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR, cameras);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        spdlog::debug("Not enough matches, wider window search!!");
        fill(mCurrentFrame.map_points.begin(), mCurrentFrame.map_points.end(), static_cast<MapPoint *>(NULL));

        nmatches = matcher.search_by_projection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR, cameras);
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
    Optimizer::PoseOptimization(&mCurrentFrame, cameras);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.feature_count; i++)
    {
        if (mCurrentFrame.map_points[i])
        {
            if (mCurrentFrame.outliers[i])
            {
                MapPoint *pMP = mCurrentFrame.map_points[i];

                mCurrentFrame.map_points[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.outliers[i]   = false;
                if (mCurrentFrame.is_fisheye_stereo && std::cmp_less(i, mCurrentFrame.keypoints.size()))
                {
                    pMP->mbTrackInView = false;
                }
                else
                {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.id;
                nmatches--;
            }
            else if (mCurrentFrame.map_points[i]->Observations() > 0)
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
    return _use_optical_flow && !mbOnlyTracking && mState == OK && mbVelocity && !frame::initial_computations && frame::next_id > mnLastRelocFrameId + 2 &&
           frame::next_id - last_feature_frame_id_ < static_cast<unsigned long>(std::max(2, mMaxFrames / 4)) && timestamp > mLastFrame.timestamp && timestamp - mLastFrame.timestamp < 1.0 &&
           !_previous_gray.empty() && _previous_gray_frame_id == mLastFrame.id && _previous_gray.size() == mImGray.size() && mImGray.type() == CV_8UC1;
}

void Tracking::create_frame(double timestamp, keypoint_extractor &extractor)
{
    use_initial_extractor_ = &extractor == initial_keypoint_extractor.get();
    const bool inertial    = mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD;
    const bool stereo      = mSensor == System::STEREO || mSensor == System::IMU_STEREO;
    mCurrentFrame          = frame(timestamp,
                          place_vocabulary_ptr,
                          mK,
                          mDistCoef,
                          mbf,
                          mThDepth,
                          inertial ? &mLastFrame : nullptr,
                          inertial ? *mpImuCalib : IMU::Calib(),
                          stereo && is_fisheye_stereo(cameras) ? mTlr : Sophus::SE3f());

    extract_features(mCurrentFrame, extractor, mImGray, cameras, stereo ? mImRight : cv::Mat(), stereo ? right_keypoint_extractor.get() : nullptr);
}

void Tracking::extract_features(frame &frame, keypoint_extractor &extractor, const cv::Mat &left, const camera_vector &cameras, const cv::Mat &right, keypoint_extractor *right_extractor)
{
    CV_Assert(!left.empty());
    CV_Assert(right.empty() || (right_extractor && right_extractor != &extractor));
    CV_Assert(cameras.size() == (right.empty() ? 1u : 2u));

    const auto extraction_started     = std::chrono::steady_clock::now();
    frame.scale_levels                = extractor.get_levels();
    frame.scale_factor                = extractor.get_scale_factor();
    frame.log_scale_factor            = std::log(frame.scale_factor);
    frame.scale_factors               = extractor.get_scale_factors();
    frame.inverse_scale_factors       = extractor.get_inverse_scale_factors();
    frame.level_sigma_squared         = extractor.get_scale_sigma_squares();
    frame.inverse_level_sigma_squared = extractor.get_inverse_scale_sigma_squares();
    frame.keypoints.clear();
    frame.right_keypoints.clear();

    const auto extract = [](keypoint_extractor &feature_extractor, const cv::Mat &image, std::vector<keypoint> &keys, std::vector<int> lapping)
    {
        cv::Mat descriptors;
        return feature_extractor(image, cv::Mat(), keys, descriptors, lapping);
    };
    if (right.empty())
    {
        frame.mono_left  = extract(extractor, left, frame.keypoints, {0, 1000});
        frame.mono_right = -1;
    }
    else
    {
        auto left_worker = std::async(
            std::launch::async,
            [&] { return extract(extractor, left, frame.keypoints, is_fisheye_stereo(cameras) ? static_cast<KannalaBrandt8 &>(*cameras[0]).mvLappingArea : std::vector<int>{0, 0}); });
        frame.mono_right =
            extract(*right_extractor, right, frame.right_keypoints, is_fisheye_stereo(cameras) ? static_cast<KannalaBrandt8 &>(*cameras[1]).mvLappingArea : std::vector<int>{0, 0});
        frame.mono_left = left_worker.get();
    }
    frame.extraction_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - extraction_started).count();
    frame.initialize_features(left, cameras, right, extractor.mvImagePyramid, right.empty() ? std::vector<cv::Mat>{} : right_extractor->mvImagePyramid);
}

void Tracking::extract_current_features()
{
    if (!mCurrentFrame.flow_only)
    {
        return;
    }
    const bool stereo    = mSensor == System::STEREO || mSensor == System::IMU_STEREO;
    auto      &extractor = use_initial_extractor_ ? *initial_keypoint_extractor : *left_keypoint_extractor;
    extract_features(mCurrentFrame, extractor, mImGray, cameras, stereo ? mImRight : cv::Mat(), stereo ? right_keypoint_extractor.get() : nullptr);
    last_feature_frame_id_ = mCurrentFrame.id;
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
    if (!_use_optical_flow || _previous_gray.empty() || mImGray.empty() || _previous_gray_frame_id != mLastFrame.id || _previous_gray.size() != mImGray.size() || mImGray.type() != CV_8UC1)
    {
        return 0;
    }
    if (mLastFrame.is_fisheye_stereo && (previous_right_gray_.empty() || mImRight.empty() || previous_right_gray_.size() != mImRight.size() || mImRight.type() != CV_8UC1))
    {
        return 0;
    }

    const std::size_t previous_keypoint_count = mLastFrame.keypoints.size() + (mLastFrame.is_fisheye_stereo ? mLastFrame.right_keypoints.size() : 0);
    const std::size_t current_keypoint_count  = mCurrentFrame.keypoints.size() + (mCurrentFrame.is_fisheye_stereo ? mCurrentFrame.right_keypoints.size() : 0);

    if (previous_keypoint_count != static_cast<std::size_t>(mLastFrame.feature_count) || current_keypoint_count != static_cast<std::size_t>(mCurrentFrame.feature_count) ||
        mLastFrame.feature_count != static_cast<int>(mLastFrame.map_points.size()) || mLastFrame.feature_count != static_cast<int>(mLastFrame.outliers.size()) ||
        mCurrentFrame.feature_count != static_cast<int>(mCurrentFrame.map_points.size()))
    {
        return 0;
    }

    std::array<std::vector<keypoint>, 2>   tracked_keys;
    std::array<std::vector<MapPoint *>, 2> tracked_points;
    const int                              view_count = mLastFrame.is_fisheye_stereo ? 2 : 1;
    for (int view = 0; view < view_count; ++view)
    {
        const bool                     right          = view == 1;
        const auto                    &previous_image = right ? previous_right_gray_ : _previous_gray;
        const auto                    &current_image  = right ? mImRight : mImGray;
        const auto                    &keys           = right ? mLastFrame.right_keypoints : mLastFrame.keypoints;
        const auto                     offset         = right ? mLastFrame.keypoints.size() : 0;
        const auto                     camera_pose    = right ? mCurrentFrame.get_relative_pose_trl() * mCurrentFrame.get_pose() : mCurrentFrame.get_pose();
        std::vector<cv::Point2f>       previous_points;
        std::vector<cv::Point2f>       current_points;
        std::vector<keypoint>          previous_keys;
        std::vector<MapPoint *>        map_points;
        std::array<int, 16>            cell_counts{};
        std::unordered_set<MapPoint *> selected_points;
        previous_points.reserve(keys.size());
        map_points.reserve(keys.size());

        for (std::size_t index = 0; index < keys.size(); ++index)
        {
            MapPoint *map_point = mLastFrame.map_points[offset + index];

            if (!map_point || mLastFrame.outliers[offset + index] || map_point->isBad() || map_point->Observations() == 0)
            {
                continue;
            }

            const auto &key = keys[index];
            if (!std::isfinite(key.pt.x) || !std::isfinite(key.pt.y) || key.pt.x < 0 || key.pt.y < 0 || key.pt.x >= previous_image.cols || key.pt.y >= previous_image.rows)
            {
                continue;
            }
            const int x          = static_cast<int>(key.pt.x * 4 / previous_image.cols);
            const int y          = static_cast<int>(key.pt.y * 4 / previous_image.rows);
            auto     &cell_count = cell_counts[static_cast<std::size_t>(y * 4 + x)];
            if (cell_count >= 64 || !selected_points.insert(map_point).second)
            {
                continue;
            }
            ++cell_count;
            previous_points.push_back(key.pt);
            previous_keys.push_back(key);
            map_points.push_back(map_point);
            cv::Point2f prediction = key.pt;
            // Camera-model projections are in LK coordinates except for distorted pinhole input.
            if (is_fisheye_stereo(cameras) || mDistCoef.at<float>(0) == 0.0f)
            {
                const Eigen::Vector3f position = camera_pose * map_point->GetWorldPos();
                if (position.z() > 0)
                {
                    const Eigen::Vector2f projected = cameras[view]->project(position);
                    if (projected.allFinite() && projected.x() >= 0 && projected.y() >= 0 && projected.x() < current_image.cols && projected.y() < current_image.rows)
                    {
                        prediction = cv::Point2f(projected.x(), projected.y());
                    }
                }
            }
            current_points.push_back(prediction);
        }

        if (previous_points.empty())
        {
            continue;
        }
        flow_reference_matches_ += static_cast<int>(previous_points.size());

        std::vector<unsigned char> forward_status;
        std::vector<float>         forward_error;
        const cv::Size             window_size(21, 21);
        const cv::TermCriteria     criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        const auto                 flow_started     = std::chrono::steady_clock::now();
        const auto                &previous_pyramid = mLastFrame.get_optical_flow_pyramid(previous_image, right);
        const auto                &current_pyramid  = mCurrentFrame.get_optical_flow_pyramid(current_image, right);
        cv::calcOpticalFlowPyrLK(previous_pyramid, current_pyramid, previous_points, current_points, forward_status, forward_error, window_size, 3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

        std::vector<cv::Point2f>   backward_points;
        std::vector<unsigned char> backward_status;
        std::vector<float>         backward_error;
        cv::calcOpticalFlowPyrLK(current_pyramid, previous_pyramid, current_points, backward_points, backward_status, backward_error, window_size, 3, criteria);
        mCurrentFrame.optical_flow_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - flow_started).count();

        tracked_keys[view].reserve(previous_points.size());
        tracked_points[view].reserve(previous_points.size());

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

            if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0.0f || point.x >= current_image.cols || point.y < 0.0f || point.y >= current_image.rows)
            {
                continue;
            }

            auto key       = previous_keys[index];
            key.pt         = point;
            key.display.pt = point;
            tracked_keys[view].push_back(key);
            tracked_points[view].push_back(map_points[index]);
        }
    }

    mCurrentFrame.flow_candidates = flow_reference_matches_;
    mCurrentFrame.set_tracked_features(std::move(tracked_keys[0]), std::move(tracked_points[0]), std::move(tracked_keys[1]), std::move(tracked_points[1]));
    mCurrentFrame.flow_tracks = mCurrentFrame.feature_count;
    return mCurrentFrame.feature_count >= 0.6 * flow_reference_matches_ ? mCurrentFrame.feature_count : 0;
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
    for (int i = 0; i < mCurrentFrame.feature_count; i++)
    {
        if (mCurrentFrame.map_points[i])
        {
            aux1++;
            if (mCurrentFrame.outliers[i])
            {
                aux2++;
            }
        }
    }

    if (!mpAtlas->is_imu_initialized())
    {
        Optimizer::PoseOptimization(&mCurrentFrame, cameras);
    }
    else
    {
        if (mCurrentFrame.id <= mnLastRelocFrameId + mnFramesToResetIMU)
        {
            spdlog::debug("TLM: PoseOptimization ");
            Optimizer::PoseOptimization(&mCurrentFrame, cameras);
        }
        else
        {
            // A flow recovery can reoptimize this frame after the last-frame prior was consumed.
            delete mCurrentFrame.imu_constraint;
            mCurrentFrame.imu_constraint = nullptr;
            if (!mbMapUpdated && mLastFrame.imu_constraint)
            {
                spdlog::trace("TLM: PoseInertialOptimizationLastFrame ");
                Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame, cameras);
            }
            else
            {
                spdlog::trace("TLM: PoseInertialOptimizationLastKeyFrame ");
                Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame, cameras);
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for (int i = 0; i < mCurrentFrame.feature_count; i++)
    {
        if (mCurrentFrame.map_points[i])
        {
            aux1++;
            if (mCurrentFrame.outliers[i])
            {
                aux2++;
            }
        }
    }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.feature_count; i++)
    {
        if (mCurrentFrame.map_points[i])
        {
            if (!mCurrentFrame.outliers[i])
            {
                mCurrentFrame.map_points[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.map_points[i]->Observations() > 0)
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
                mCurrentFrame.map_points[i] = static_cast<MapPoint *>(NULL);
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
    if (mCurrentFrame.id < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
    {
        return false;
    }

    if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
    {
        return true;
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        if ((mnMatchesInliers < 15 && mpAtlas->is_imu_initialized()) || (mnMatchesInliers < 50 && !mpAtlas->is_imu_initialized()))
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
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->get_current_map()->is_imu_initialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.timestamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
        {
            return true;
        }
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.timestamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
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

    const int nKFs = mpAtlas->key_frames_in_map();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.id < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
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
    // Flow tracks a bounded subset, not the full extracted feature population.
    if (mCurrentFrame.flow_only)
    {
        nRefMatches = std::min(nRefMatches, flow_reference_matches_);
    }

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose    = 0;

    if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
    {
        int N = mCurrentFrame.is_fisheye_stereo ? static_cast<int>(mCurrentFrame.keypoints.size()) : mCurrentFrame.feature_count;
        for (int i = 0; i < N; i++)
        {
            if (mCurrentFrame.depths[i] > 0 && mCurrentFrame.depths[i] < mThDepth)
            {
                if (mCurrentFrame.map_points[i] && !mCurrentFrame.outliers[i])
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
    float thRefRatio = 0.75f;
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

    if (is_fisheye_stereo(cameras))
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
    const bool c1a = mCurrentFrame.id >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.id >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO && mSensor != System::IMU_RGBD &&
                     (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) && mnMatchesInliers > 15);

    //  Temporal condition for Inertial cases
    bool c3 = false;
    if (mpLastKeyFrame)
    {
        if (mSensor == System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.timestamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
            {
                c3 = true;
            }
        }
        else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.timestamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
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
    if (mpLocalMapper->IsInitializing() && !mpAtlas->is_imu_initialized())
    {
        return;
    }
    if (mCurrentFrame.flow_only)
    {
        const frame flow_frame(mCurrentFrame);
        extract_current_features();
        descriptor_matcher matcher(0.9, true);
        matcher.search_by_projection(mCurrentFrame, mLastFrame, mSensor == System::STEREO ? 7 : 15, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR, cameras);
        SearchLocalPoints();
        const int inliers = Optimizer::PoseOptimization(&mCurrentFrame, cameras);
        if (inliers < 50)
        {
            spdlog::warn("Frame {}: postponing keyframe; only {} descriptor inliers after feature refresh", mCurrentFrame.id, inliers);
            mCurrentFrame = flow_frame;
            return;
        }
        mnMatchesInliers = inliers;
        mVelocity        = mCurrentFrame.get_pose() * mLastFrame.get_pose().inverse();
        if (!NeedNewKeyFrame())
        {
            return;
        }
    }

    if (!mpLocalMapper->SetNotStop(true))
    {
        return;
    }

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->get_current_map(), mpKeyFrameDB, cameras);

    if (mpAtlas->is_imu_initialized())
    {
        pKF->bImu = true;
    }

    pKF->SetNewBias(mCurrentFrame.imu_bias);
    mpReferenceKF                     = pKF;
    mCurrentFrame.reference_key_frame = pKF;

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
        mCurrentFrame.update_pose_matrices();
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            maxPoint = 100;
        }

        vector<pair<float, int>> vDepthIdx;
        int                      N = mCurrentFrame.is_fisheye_stereo ? static_cast<int>(mCurrentFrame.keypoints.size()) : mCurrentFrame.feature_count;
        vDepthIdx.reserve(mCurrentFrame.feature_count);
        for (int i = 0; i < N; i++)
        {
            float z = mCurrentFrame.depths[i];
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
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.map_points[i];
                if (!pMP)
                {
                    bCreateNew = true;
                }
                else if (pMP->Observations() < 1)
                {
                    bCreateNew                  = true;
                    mCurrentFrame.map_points[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if (!mCurrentFrame.is_fisheye_stereo)
                    {
                        mCurrentFrame.unproject_stereo(i, x3D);
                    }
                    else
                    {
                        x3D = mCurrentFrame.unproject_stereo_fisheye(i);
                    }

                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->get_current_map());
                    pNewMP->AddObservation(pKF, i);

                    // Check if it is a stereo observation in order to not
                    // duplicate mappoints
                    if (mCurrentFrame.is_fisheye_stereo && mCurrentFrame.left_to_right_matches[i] >= 0)
                    {
                        mCurrentFrame.map_points[static_cast<int>(mCurrentFrame.keypoints.size()) + mCurrentFrame.left_to_right_matches[i]] = pNewMP;
                        pNewMP->AddObservation(pKF, static_cast<int>(mCurrentFrame.keypoints.size()) + mCurrentFrame.left_to_right_matches[i]);
                        pKF->AddMapPoint(pNewMP, static_cast<int>(mCurrentFrame.keypoints.size()) + mCurrentFrame.left_to_right_matches[i]);
                    }

                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->add_map_point(pNewMP);

                    mCurrentFrame.map_points[i] = pNewMP;
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

    mnLastKeyFrameId = mCurrentFrame.id;
    mpLastKeyFrame   = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.map_points.begin(), vend = mCurrentFrame.map_points.end(); vit != vend; vit++)
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
                pMP->mnLastFrameSeen = mCurrentFrame.id;
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

        if (pMP->mnLastFrameSeen == mCurrentFrame.id)
        {
            continue;
        }
        if (pMP->isBad())
        {
            continue;
        }
        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.is_in_frustum(pMP, 0.5, cameras))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        if (pMP->mbTrackInView)
        {
            mCurrentFrame.projected_points[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
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
        if (mpAtlas->is_imu_initialized())
        {
            if (mpAtlas->get_current_map()->get_inertial_ba2())
            {
                th = 2;
            }
            else
            {
                th = 6;
            }
        }
        else if (!mpAtlas->is_imu_initialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th = 10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.id < mnLastRelocFrameId + 2)
        {
            th = 5;
        }

        if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
        {
            th = 15; // 15
        }

        matcher.search_by_projection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->set_reference_map_points(mvpLocalMapPoints);

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
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.id)
            {
                continue;
            }
            if (!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.id;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    std::map<KeyFrame *, int> keyframeCounter;
    if (!mpAtlas->is_imu_initialized() || (mCurrentFrame.id < mnLastRelocFrameId + 2))
    {
        for (int i = 0; i < mCurrentFrame.feature_count; i++)
        {
            MapPoint *pMP = mCurrentFrame.map_points[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    const std::map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                    for (std::map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    {
                        keyframeCounter[it->first]++;
                    }
                }
                else
                {
                    mCurrentFrame.map_points[i] = NULL;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < mLastFrame.feature_count; i++)
        {
            // Using lastframe since current frame has not matches yet
            if (mLastFrame.map_points[i])
            {
                MapPoint *pMP = mLastFrame.map_points[i];
                if (!pMP)
                {
                    continue;
                }
                if (!pMP->isBad())
                {
                    const std::map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                    for (std::map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    {
                        keyframeCounter[it->first]++;
                    }
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.map_points[i] = NULL;
                }
            }
        }
    }

    int       max    = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (std::map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
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
        pKF->mnTrackReferenceForFrame = mCurrentFrame.id;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80) // 80
        {
            break;
        }

        KeyFrame *pKF = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.id)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.id;
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
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.id)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.id;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.id)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.id;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mvpLocalKeyFrames.size() < 80)
    {
        KeyFrame *tempKeyFrame = mCurrentFrame.last_key_frame;

        const int Nd = 20;
        for (int i = 0; i < Nd; i++)
        {
            if (!tempKeyFrame)
            {
                break;
            }
            if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.id)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.id;
                tempKeyFrame                           = tempKeyFrame->mPrevKF;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF                     = pKFmax;
        mCurrentFrame.reference_key_frame = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    extract_current_features();
    spdlog::debug("Starting relocalization");
    // Compute Bag of Words Vector
    mCurrentFrame.compute_bow();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->get_current_map());

    if (vpCandidateKFs.empty())
    {
        spdlog::debug("There are not candidates");
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    descriptor_matcher matcher(0.75, true);

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
            if (!map_point || map_point->isBad() || j >= mCurrentFrame.keypoints.size())
            {
                continue;
            }
            const cv::KeyPoint   &keypoint = mCurrentFrame.keypoints[j].undistorted();
            const Eigen::Vector3f position = map_point->GetWorldPos();
            world_points.emplace_back(position.x(), position.y(), position.z());
            image_points.push_back(keypoint.pt);
            max_errors.push_back(mCurrentFrame.level_sigma_squared[keypoint.octave] * 5.991f);
        }

        vector<bool>    pnp_inliers;
        Eigen::Matrix4f Tcw_matrix;
        const int       min_inliers = std::max(10, static_cast<int>(world_points.size() * 0.5));
        if (!geometric_tools::solve_pnp_ransac(world_points, image_points, max_errors, *cameras[0], min_inliers, Tcw_matrix, pnp_inliers))
        {
            continue;
        }

        mCurrentFrame.set_pose(Sophus::SE3f(Tcw_matrix));
        set<MapPoint *> sFound;
        std::fill(mCurrentFrame.map_points.begin(), mCurrentFrame.map_points.end(), nullptr);
        size_t correspondence_index = 0;
        for (size_t j = 0; j < vvpMapPointMatches[i].size(); ++j)
        {
            MapPoint *map_point = vvpMapPointMatches[i][j];
            if (!map_point || map_point->isBad() || j >= mCurrentFrame.keypoints.size())
            {
                continue;
            }
            if (pnp_inliers[correspondence_index])
            {
                mCurrentFrame.map_points[j] = map_point;
                sFound.insert(map_point);
            }
            ++correspondence_index;
        }

        int nGood = Optimizer::PoseOptimization(&mCurrentFrame, cameras);

        if (nGood < 10)
        {
            continue;
        }

        for (int io = 0; io < mCurrentFrame.feature_count; io++)
        {
            if (mCurrentFrame.outliers[io])
            {
                mCurrentFrame.map_points[io] = static_cast<MapPoint *>(NULL);
            }
        }

        // If few inliers, search by projection in a coarse window and optimize again
        if (nGood < 50)
        {
            int nadditional = matcher2.search_by_projection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100, cameras);

            if (nadditional + nGood >= 50)
            {
                nGood = Optimizer::PoseOptimization(&mCurrentFrame, cameras);

                // If many inliers but still not enough, search by projection again in a narrower window
                // the camera has been already optimized with many points
                if (nGood > 30 && nGood < 50)
                {
                    sFound.clear();
                    for (int ip = 0; ip < mCurrentFrame.feature_count; ip++)
                    {
                        if (mCurrentFrame.map_points[ip])
                        {
                            sFound.insert(mCurrentFrame.map_points[ip]);
                        }
                    }
                    nadditional = matcher2.search_by_projection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64, cameras);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame, cameras);

                        for (int io = 0; io < mCurrentFrame.feature_count; io++)
                        {
                            if (mCurrentFrame.outliers[io])
                            {
                                mCurrentFrame.map_points[io] = NULL;
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
        mnLastRelocFrameId = mCurrentFrame.id;
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
    mpAtlas->clear_atlas();
    mpAtlas->create_new_map();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
    {
        mpAtlas->set_inertial_sensor();
    }
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    frame::next_id    = 0;
    mState            = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit             = false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame      = frame();
    mnLastRelocFrameId = 0;
    mLastFrame         = frame();
    _previous_gray.release();
    previous_right_gray_.release();
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
    map *pMap = mpAtlas->get_current_map();

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
    mpAtlas->clear_map();

    // KeyFrame::nNextId = mpAtlas->get_last_init_kf_id();
    // frame::next_id = mnLastInitFrameId;
    mnLastInitFrameId = frame::next_id;
    mState            = NO_IMAGES_YET; // NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    spdlog::debug("First frame ID: {}", mnFirstFrameId);
    for (map *pMap : mpAtlas->get_all_maps())
    {
        if (pMap->get_all_key_frames().size() > 0)
        {
            if (index > pMap->get_lowest_key_frame_id())
            {
                index = pMap->get_lowest_key_frame_id();
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

    mlbLost = lbLost;

    mnInitialFrameId   = mCurrentFrame.id;
    mnLastRelocFrameId = mCurrentFrame.id;

    mCurrentFrame = frame();
    mLastFrame    = frame();
    _previous_gray.release();
    previous_right_gray_.release();
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

void Tracking::InformOnlyTracking(const bool &flag) { mbOnlyTracking = flag; }

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
{
    map                                  *pMap = pCurrentKeyFrame->GetMap();
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

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.set_new_bias(mLastBias);
    mCurrentFrame.set_new_bias(mLastBias);

    while (!mCurrentFrame.imu_is_preintegrated())
    {
        usleep(500);
    }

    if (mLastFrame.id == mLastFrame.last_key_frame->mnFrameId)
    {
        mLastFrame.set_imu_pose_velocity(mLastFrame.last_key_frame->GetImuRotation(), mLastFrame.last_key_frame->GetImuPosition(), mLastFrame.last_key_frame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.last_key_frame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.last_key_frame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.last_key_frame->GetVelocity();
        float                 t12  = mLastFrame.imu_preintegrated->dT;

        mLastFrame.set_imu_pose_velocity(IMU::NormalizeRotation(Rwb1 * mLastFrame.imu_preintegrated->GetUpdatedDeltaRotation()),
                                         twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.imu_preintegrated->GetUpdatedDeltaPosition(),
                                         Vwb1 + Gz * t12 + Rwb1 * mLastFrame.imu_preintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.imu_preintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.last_key_frame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.last_key_frame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.last_key_frame->GetVelocity();
        float                 t12  = mCurrentFrame.imu_preintegrated->dT;

        mCurrentFrame.set_imu_pose_velocity(IMU::NormalizeRotation(Rwb1 * mCurrentFrame.imu_preintegrated->GetUpdatedDeltaRotation()),
                                            twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.imu_preintegrated->GetUpdatedDeltaPosition(),
                                            Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.imu_preintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.id;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
    _previous_gray.release();
    previous_right_gray_.release();
}

int Tracking::GetNumberDataset() { return mnNumDataset; }

int Tracking::GetMatchesInliers() { return mnMatchesInliers; }

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    // mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, map *pMap)
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
