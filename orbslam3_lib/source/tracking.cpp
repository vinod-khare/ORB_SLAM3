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

#include "tracking.h"

#include "G2oTypes.h"
#include "KannalaBrandt8.h"
#include "optimizer.h"
#include "pinhole.h"
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

tracking::tracking(
    system *pSys, place_vocabulary *pVoc, atlas *pAtlas, key_frame_database *pKFDB, const int sensor_type, settings *settings, const camera_vector &camera_list, const string &_nameSeq)
    : state(NO_IMAGES_YET), sensor(sensor_type), tracked_frame_count(0), step_by_step_enabled(false), only_tracking(false), map_updated(false), visual_odometry(false),
      place_vocabulary_ptr(pVoc), database(pKFDB), ready_to_initialize(false), owner_system(pSys), step_by_step(false), owner_atlas(pAtlas), last_key_frame(nullptr),
      last_relocalization_frame_id(0), time_recently_lost(5.0), first_frame_id(0), initial_frame_id(0), map_created(false), cameras(camera_list)
{
    load_parameters(settings);

    initialization_id                = 0;
    last_id                          = 0;
    initialize_with_three_key_frames = false;
    dataset_count                    = 0;

    for (const auto &pCam : cameras)
    {
        const char *camera_type = "unknown";
        if (pCam->get_type() == geometric_camera::cam_pinhole)
        {
            camera_type = "pinhole";
        }
        else if (pCam->get_type() == geometric_camera::cam_fisheye)
        {
            camera_type = "fisheye";
        }
        spdlog::debug("Camera {} is {}", pCam->get_id(), camera_type);
    }
}

tracking::~tracking() {}

// TODO: don't use raw pointer
void tracking::load_parameters(settings *settings)
{

    if (settings->need_to_undistort())
    {
        distortion_coefficients = settings->camera_1_distortion_coefficients();
    }
    else
    {
        distortion_coefficients = cv::Mat::zeros(4, 1, CV_32F);
    }

    // TODO: missing image scaling and rectification
    image_scale = 1.0f;

    camera_matrix                 = cv::Mat::eye(3, 3, CV_32F);
    camera_matrix.at<float>(0, 0) = cameras[0]->get_parameter(0);
    camera_matrix.at<float>(1, 1) = cameras[0]->get_parameter(1);
    camera_matrix.at<float>(0, 2) = cameras[0]->get_parameter(2);
    camera_matrix.at<float>(1, 2) = cameras[0]->get_parameter(3);

    camera_matrix_eigen.setIdentity();
    camera_matrix_eigen(0, 0) = cameras[0]->get_parameter(0);
    camera_matrix_eigen(1, 1) = cameras[0]->get_parameter(1);
    camera_matrix_eigen(0, 2) = cameras[0]->get_parameter(2);
    camera_matrix_eigen(1, 2) = cameras[0]->get_parameter(3);

    if ((sensor == system::stereo || sensor == system::imu_stereo || sensor == system::imu_rgbd) && settings->camera_type() == settings::kannala_brandt)
    {

        left_to_right_pose = settings->left_to_right_pose();
    }

    if (sensor == system::stereo || sensor == system::rgbd || sensor == system::imu_stereo || sensor == system::imu_rgbd)
    {
        baseline_fx     = settings->baseline_fx();
        depth_threshold = settings->baseline() * settings->depth_threshold();
    }

    if (sensor == system::rgbd || sensor == system::imu_rgbd)
    {
        depth_map_factor = settings->depth_map_factor();
        if (fabs(depth_map_factor) < 1e-5)
        {
            depth_map_factor = 1;
        }
        else
        {
            depth_map_factor = 1.0f / depth_map_factor;
        }
    }

    minimum_frames      = 0;
    maximum_frames      = settings->fps();
    frames_to_reset_imu = maximum_frames;
    rgb                 = settings->rgb();

    // ORB parameters
    int                          nFeatures    = settings->feature_count();
    int                          nLevels      = settings->level_count();
    int                          fIniThFAST   = settings->initial_fast_threshold();
    int                          fMinThFAST   = settings->minimum_fast_threshold();
    float                        fScaleFactor = settings->scale_factor();
    const keypoint_detector_type detector     = settings->keypoint_detector();
    const auto                   descriptor   = settings->keypoint_descriptor();
    descriptor_matcher::set_backend(settings->matcher_backend());
    use_optical_flow = settings->use_optical_flow() && sensor != system::rgbd && sensor != system::imu_rgbd;

    spdlog::info("Sparse optical flow: {}", use_optical_flow ? "enabled" : "disabled");

    left_keypoint_extractor = std::make_unique<keypoint_extractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);
    if (place_vocabulary_ptr && (place_vocabulary_ptr->descriptor_type() != left_keypoint_extractor->get_descriptor_type() ||
                                 place_vocabulary_ptr->descriptor_size() != left_keypoint_extractor->get_descriptor_size()))
    {
        throw std::runtime_error("The place-recognition vocabulary descriptor format does not match keypoint.descriptor.type");
    }

    if (sensor == system::stereo || sensor == system::imu_stereo)
    {
        right_keypoint_extractor = std::make_unique<keypoint_extractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);
    }

    if (sensor == system::monocular || sensor == system::imu_monocular)
    {
        initial_keypoint_extractor = std::make_unique<keypoint_extractor>(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST, detector, descriptor);
    }

    // IMU parameters
    Sophus::SE3f Tbc            = settings->imu_to_camera_pose();
    insert_key_frames_when_lost = settings->insert_key_frames_when_lost();
    imu_frequency               = settings->imu_frequency();
    imu_period                  = 0.001; // TODO: ESTO ESTA BIEN?
    float Ng                    = settings->gyroscope_noise();
    float Na                    = settings->accelerometer_noise();
    float Ngw                   = settings->gyroscope_random_walk();
    float Naw                   = settings->accelerometer_random_walk();

    const float sf  = sqrt(imu_frequency);
    imu_calibration = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

    imu_preintegrated_from_last_key_frame = new IMU::Preintegrated(IMU::Bias(), *imu_calibration);
}

void tracking::set_local_mapper(local_mapping *pLocalMapper) { local_mapper = pLocalMapper; }

void tracking::set_loop_closing(loop_closing *pLoopClosing) { loop_closer = pLoopClosing; }

void tracking::set_reset_hooks(std::function<void()> on_reset_start, std::function<void()> on_reset_complete)
{
    reset_start_callback    = std::move(on_reset_start);
    reset_complete_callback = std::move(on_reset_complete);
}

void tracking::set_step_by_step(bool bSet) { step_by_step = bSet; }

bool tracking::get_step_by_step() { return step_by_step; }

void tracking::set_frame_colors(const cv::Mat &left, const cv::Mat &right)
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
            cv::cvtColor(image, result, rgb ? cv::COLOR_RGBA2BGR : cv::COLOR_BGRA2BGR);
        }
        else if (rgb)
        {
            cv::cvtColor(image, result, cv::COLOR_RGB2BGR);
        }
        else
        {
            result = image.clone();
        }
        return result;
    };
    color_image                     = next_frame_color.empty() ? to_bgr(left) : std::move(next_frame_color);
    current_frame.color_image       = color_image;
    current_frame.color_image_right = next_frame_color_right.empty() ? to_bgr(right) : std::move(next_frame_color_right);
    current_frame.color_is_rgb      = false;
    next_frame_color.release();
    next_frame_color_right.release();
}

Sophus::SE3f tracking::grab_image_stereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{

    gray_image          = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    right_image         = imRectRight;

    if (gray_image.channels() == 3)
    {
        if (rgb)
        {
            cvtColor(gray_image, gray_image, cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
        }
    }
    else if (gray_image.channels() == 4)
    {
        if (rgb)
        {
            cvtColor(gray_image, gray_image, cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(gray_image, gray_image, cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGRA2GRAY);
        }
    }

    right_image = imGrayRight;
    if (can_defer_features(timestamp))
    {
        current_frame = frame(last_frame, timestamp);
    }
    else
    {
        create_frame(timestamp, *left_keypoint_extractor);
    }

    set_frame_colors(imRectLeft, imRectRight);
    current_frame.name_file = filename;
    current_frame.dataset   = dataset_count;

    current_frame.optical_flow_pyramid.swap(next_optical_flow_pyramid);
    if (!current_frame.flow_only)
    {
        last_feature_frame_id = current_frame.id;
        ++feature_frames;
    }
    track();
    if (current_frame.flow_only && state == OK)
    {
        ++flow_frames;
    }

    if (use_optical_flow && last_frame.id == current_frame.id && !gray_image.empty())
    {
        previous_gray          = gray_image.clone();
        previous_right_gray    = right_image.clone();
        previous_gray_frame_id = current_frame.id;
    }
    else
    {
        previous_gray.release();
        previous_right_gray.release();
    }

    return current_frame.get_pose();
}

Sophus::SE3f tracking::grab_image_rgbd(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename)
{
    gray_image      = imRGB;
    cv::Mat imDepth = imD;

    if (gray_image.channels() == 3)
    {
        if (rgb)
        {
            cvtColor(gray_image, gray_image, cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
        }
    }
    else if (gray_image.channels() == 4)
    {
        if (rgb)
        {
            cvtColor(gray_image, gray_image, cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(gray_image, gray_image, cv::COLOR_BGRA2GRAY);
        }
    }

    if ((fabs(depth_map_factor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
    {
        imDepth.convertTo(imDepth, CV_32F, depth_map_factor);
    }

    create_frame(timestamp, *left_keypoint_extractor);
    current_frame.compute_stereo_from_rgbd(imDepth);

    set_frame_colors(imRGB);
    current_frame.name_file = filename;
    current_frame.dataset   = dataset_count;

    track();

    return current_frame.get_pose();
}

Sophus::SE3f tracking::grab_image_monocular(const cv::Mat &im, const double &timestamp, string filename)
{
    gray_image = im;

    if (gray_image.channels() == 3)
    {
        if (rgb)
        {
            cvtColor(gray_image, gray_image, cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
        }
    }
    else if (gray_image.channels() == 4)
    {
        if (rgb)
        {
            cvtColor(gray_image, gray_image, cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(gray_image, gray_image, cv::COLOR_BGRA2GRAY);
        }
    }

    if (can_defer_features(timestamp))
    {
        current_frame = frame(last_frame, timestamp);
    }
    else
    {
        const bool initializing = state == NOT_INITIALIZED || state == NO_IMAGES_YET || (sensor == system::monocular && (last_id - initialization_id) < maximum_frames);
        create_frame(timestamp, initializing ? *initial_keypoint_extractor : *left_keypoint_extractor);
    }

    if (state == NO_IMAGES_YET)
    {
        first_frame_timestamp = timestamp;
    }

    set_frame_colors(im);
    current_frame.name_file = filename;
    current_frame.dataset   = dataset_count;

    last_id = current_frame.id;
    current_frame.optical_flow_pyramid.swap(next_optical_flow_pyramid);
    if (!current_frame.flow_only)
    {
        last_feature_frame_id = current_frame.id;
        ++feature_frames;
    }
    track();
    if (current_frame.flow_only && state == OK)
    {
        ++flow_frames;
    }

    if (use_optical_flow && last_frame.id == current_frame.id && !gray_image.empty())
    {
        previous_gray          = gray_image.clone();
        previous_gray_frame_id = current_frame.id;
    }
    else
    {
        previous_gray.release();
    }

    return current_frame.get_pose();
}

void tracking::grab_imu_data(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(imu_queue_mutex);
    imu_data_queue.push_back(imuMeasurement);
}

void tracking::preintegrate_imu()
{

    if (!current_frame.previous_frame)
    {
        spdlog::debug("non prev frame");
        current_frame.set_integrated();
        return;
    }

    imu_measurements_from_last_frame.clear();
    imu_measurements_from_last_frame.reserve(imu_data_queue.size());
    if (imu_data_queue.size() == 0)
    {
        spdlog::debug("Not IMU data in mlQueueImuData!!");
        current_frame.set_integrated();
        return;
    }

    while (true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(imu_queue_mutex);
            if (!imu_data_queue.empty())
            {
                IMU::Point *m = &imu_data_queue.front();
                if (m->t < current_frame.previous_frame->timestamp - imu_period)
                {
                    imu_data_queue.pop_front();
                }
                else if (m->t < current_frame.timestamp - imu_period)
                {
                    imu_measurements_from_last_frame.push_back(*m);
                    imu_data_queue.pop_front();
                }
                else
                {
                    imu_measurements_from_last_frame.push_back(*m);
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

    const int n = imu_measurements_from_last_frame.size() - 1;
    if (n == 0)
    {
        spdlog::warn("Empty IMU measurements vector");
        return;
    }

    IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(last_frame.imu_bias, current_frame.imu_calibration);

    for (int i = 0; i < n; i++)
    {
        float           tstep;
        Eigen::Vector3f acc, angVel;
        if ((i == 0) && (i < (n - 1)))
        {
            float tab  = imu_measurements_from_last_frame[i + 1].t - imu_measurements_from_last_frame[i].t;
            float tini = imu_measurements_from_last_frame[i].t - current_frame.previous_frame->timestamp;
            acc        = (imu_measurements_from_last_frame[i].a + imu_measurements_from_last_frame[i + 1].a -
                   (imu_measurements_from_last_frame[i + 1].a - imu_measurements_from_last_frame[i].a) * (tini / tab)) *
                  0.5f;
            angVel = (imu_measurements_from_last_frame[i].w + imu_measurements_from_last_frame[i + 1].w -
                      (imu_measurements_from_last_frame[i + 1].w - imu_measurements_from_last_frame[i].w) * (tini / tab)) *
                     0.5f;
            tstep = imu_measurements_from_last_frame[i + 1].t - current_frame.previous_frame->timestamp;
        }
        else if (i < (n - 1))
        {
            acc    = (imu_measurements_from_last_frame[i].a + imu_measurements_from_last_frame[i + 1].a) * 0.5f;
            angVel = (imu_measurements_from_last_frame[i].w + imu_measurements_from_last_frame[i + 1].w) * 0.5f;
            tstep  = imu_measurements_from_last_frame[i + 1].t - imu_measurements_from_last_frame[i].t;
        }
        else if ((i > 0) && (i == (n - 1)))
        {
            float tab  = imu_measurements_from_last_frame[i + 1].t - imu_measurements_from_last_frame[i].t;
            float tend = imu_measurements_from_last_frame[i + 1].t - current_frame.timestamp;
            acc        = (imu_measurements_from_last_frame[i].a + imu_measurements_from_last_frame[i + 1].a -
                   (imu_measurements_from_last_frame[i + 1].a - imu_measurements_from_last_frame[i].a) * (tend / tab)) *
                  0.5f;
            angVel = (imu_measurements_from_last_frame[i].w + imu_measurements_from_last_frame[i + 1].w -
                      (imu_measurements_from_last_frame[i + 1].w - imu_measurements_from_last_frame[i].w) * (tend / tab)) *
                     0.5f;
            tstep = current_frame.timestamp - imu_measurements_from_last_frame[i].t;
        }
        else if ((i == 0) && (i == (n - 1)))
        {
            acc    = imu_measurements_from_last_frame[i].a;
            angVel = imu_measurements_from_last_frame[i].w;
            tstep  = current_frame.timestamp - current_frame.previous_frame->timestamp;
        }

        if (!imu_preintegrated_from_last_key_frame)
        {
            spdlog::error("IMU preintegration from the last keyframe is missing");
        }
        imu_preintegrated_from_last_key_frame->IntegrateNewMeasurement(acc, angVel, tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
    }

    current_frame.frame_imu_preintegrated = pImuPreintegratedFromLastFrame;
    current_frame.imu_preintegrated       = imu_preintegrated_from_last_key_frame;
    current_frame.last_key_frame          = last_key_frame;

    current_frame.set_integrated();
}

bool tracking::predict_state_imu()
{
    if (!current_frame.previous_frame)
    {
        spdlog::trace("No last frame");
        return false;
    }

    if (map_updated && last_key_frame)
    {
        const Eigen::Vector3f twb1 = last_key_frame->get_imu_position();
        const Eigen::Matrix3f Rwb1 = last_key_frame->get_imu_rotation();
        const Eigen::Vector3f Vwb1 = last_key_frame->get_velocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float           t12 = imu_preintegrated_from_last_key_frame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * imu_preintegrated_from_last_key_frame->GetDeltaRotation(last_key_frame->get_imu_bias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * imu_preintegrated_from_last_key_frame->GetDeltaPosition(last_key_frame->get_imu_bias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * imu_preintegrated_from_last_key_frame->GetDeltaVelocity(last_key_frame->get_imu_bias());
        current_frame.set_imu_pose_velocity(Rwb2, twb2, Vwb2);

        current_frame.imu_bias           = last_key_frame->get_imu_bias();
        current_frame.predicted_imu_bias = current_frame.imu_bias;
        return true;
    }
    else if (!map_updated)
    {
        const Eigen::Vector3f twb1 = last_frame.get_imu_position();
        const Eigen::Matrix3f Rwb1 = last_frame.get_imu_rotation();
        const Eigen::Vector3f Vwb1 = last_frame.get_velocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float           t12 = current_frame.frame_imu_preintegrated->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * current_frame.frame_imu_preintegrated->GetDeltaRotation(last_frame.imu_bias));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * current_frame.frame_imu_preintegrated->GetDeltaPosition(last_frame.imu_bias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * current_frame.frame_imu_preintegrated->GetDeltaVelocity(last_frame.imu_bias);

        current_frame.set_imu_pose_velocity(Rwb2, twb2, Vwb2);

        current_frame.imu_bias           = last_frame.imu_bias;
        current_frame.predicted_imu_bias = current_frame.imu_bias;
        return true;
    }
    else
    {
        spdlog::warn("IMU prediction unavailable");
    }

    return false;
}

void tracking::reset_frame_imu()
{
    // TODO To implement...
}

void tracking::track()
{

    if (step_by_step)
    {
        spdlog::debug("Tracking is waiting for the next step");
        while (!step_by_step_enabled && step_by_step)
        {
            usleep(500);
        }
        step_by_step_enabled = false;
    }

    if (local_mapper->bad_imu)
    {
        spdlog::warn("Resetting map because local mapper set the bad IMU flag");
        owner_system->reset_active_map();
        return;
    }

    map *pCurrentMap = owner_atlas->get_current_map();
    if (!pCurrentMap)
    {
        spdlog::error("There is no active map in the Atlas");
    }

    if (state != NO_IMAGES_YET)
    {
        if (last_frame.timestamp > current_frame.timestamp)
        {
            spdlog::error("Frame with a timestamp older than the previous frame detected");
            unique_lock<mutex> lock(imu_queue_mutex);
            imu_data_queue.clear();
            create_map_in_atlas();
            return;
        }
        else if (current_frame.timestamp > last_frame.timestamp + 1.0)
        {
            if (owner_atlas->is_inertial())
            {

                if (owner_atlas->is_imu_initialized())
                {
                    spdlog::warn("Timestamp jump detected; setting state to LOST and resetting IMU integration");
                    if (!pCurrentMap->get_inertial_ba2())
                    {
                        owner_system->reset_active_map();
                    }
                    else
                    {
                        create_map_in_atlas();
                    }
                }
                else
                {
                    spdlog::warn("Timestamp jump detected before IMU initialization; resetting map");
                    owner_system->reset_active_map();
                }
                return;
            }
        }
    }

    if ((sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd) && last_key_frame)
    {
        current_frame.set_new_bias(last_key_frame->get_imu_bias());
    }

    if (state == NO_IMAGES_YET)
    {
        state = NOT_INITIALIZED;
    }

    last_processed_state = state;

    if ((sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd) && !map_created)
    {
        preintegrate_imu();
    }
    map_created = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->map_update_mutex);

    map_updated = false;

    int nCurMapChangeIndex = pCurrentMap->get_map_change_index();
    int nMapChangeIndex    = pCurrentMap->get_last_map_change();
    if (nCurMapChangeIndex > nMapChangeIndex)
    {
        pCurrentMap->set_last_map_change(nCurMapChangeIndex);
        map_updated = true;
    }

    if (state == NOT_INITIALIZED)
    {
        if (sensor == system::stereo || sensor == system::rgbd || sensor == system::imu_stereo || sensor == system::imu_rgbd)
        {
            stereo_initialization();
        }
        else
        {
            monocular_initialization();
        }

        if (state != OK) // If rightly initialized, mState=OK
        {
            last_frame = frame(current_frame);
            return;
        }

        if (owner_atlas->get_all_maps().size() == 1)
        {
            first_frame_id = current_frame.id;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK = false;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!only_tracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if (state == OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                check_replaced_in_last_frame();

                if ((!has_velocity && !pCurrentMap->is_imu_initialized()) || current_frame.id < last_relocalization_frame_id + 2)
                {
                    spdlog::trace("TRACK: Track with respect to the reference KF");
                    bOK = track_reference_key_frame();
                }
                else
                {
                    spdlog::trace("TRACK: Track with motion model");
                    bOK = track_with_motion_model();
                    if (!bOK)
                    {
                        bOK = track_reference_key_frame();
                    }
                }

                if (!bOK)
                {
                    if (current_frame.id <= (last_relocalization_frame_id + frames_to_reset_imu) &&
                        (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd))
                    {
                        state = LOST;
                    }
                    else if (pCurrentMap->key_frame_count() > 10)
                    {
                        state          = RECENTLY_LOST;
                        lost_timestamp = current_frame.timestamp;
                    }
                    else
                    {
                        state = LOST;
                    }
                }
            }
            else
            {

                if (state == RECENTLY_LOST)
                {
                    spdlog::trace("Lost for a short time");

                    bOK = true;
                    if ((sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd))
                    {
                        if (pCurrentMap->is_imu_initialized())
                        {
                            predict_state_imu();
                        }
                        else
                        {
                            bOK = false;
                        }

                        if (current_frame.timestamp - lost_timestamp > time_recently_lost)
                        {
                            state = LOST;
                            spdlog::debug("Track Lost...");
                            bOK = false;
                        }
                    }
                    else
                    {
                        // Relocalization
                        bOK = relocalization();
                        if (current_frame.timestamp - lost_timestamp > 3.0f && !bOK)
                        {
                            state = LOST;
                            spdlog::debug("Track Lost...");
                            bOK = false;
                        }
                    }
                }
                else if (state == LOST)
                {

                    spdlog::debug("A new map is started...");

                    if (pCurrentMap->key_frame_count() < 10)
                    {
                        owner_system->reset_active_map();
                        spdlog::debug("Reseting current map...");
                    }
                    else
                    {
                        create_map_in_atlas();
                    }

                    if (last_key_frame)
                    {
                        last_key_frame = static_cast<key_frame *>(NULL);
                    }

                    spdlog::debug("done");

                    return;
                }
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if (state == LOST)
            {
                if (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd)
                {
                    spdlog::debug("IMU. State LOST");
                }
                bOK = relocalization();
            }
            else
            {
                if (!visual_odometry)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if (has_velocity)
                    {
                        bOK = track_with_motion_model();
                    }
                    else
                    {
                        bOK = track_reference_key_frame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool                bOKMM    = false;
                    bool                bOKReloc = false;
                    vector<map_point *> vpMPsMM;
                    vector<bool>        vbOutMM;
                    Sophus::SE3f        TcwMM;
                    if (has_velocity)
                    {
                        bOKMM   = track_with_motion_model();
                        vpMPsMM = current_frame.map_points;
                        vbOutMM = current_frame.outliers;
                        TcwMM   = current_frame.get_pose();
                    }
                    bOKReloc = relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        current_frame.set_pose(TcwMM);
                        current_frame.map_points = vpMPsMM;
                        current_frame.outliers   = vbOutMM;

                        if (visual_odometry)
                        {
                            for (int i = 0; i < current_frame.feature_count; i++)
                            {
                                if (current_frame.map_points[i] && !current_frame.outliers[i])
                                {
                                    current_frame.map_points[i]->increase_found();
                                }
                            }
                        }
                    }
                    else if (bOKReloc)
                    {
                        visual_odometry = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if (!current_frame.reference_key_frame)
        {
            current_frame.reference_key_frame = reference_key_frame;
        }

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (!only_tracking)
        {
            if (bOK)
            {
                bOK = track_local_map();
            }
            if (!bOK)
            {
                spdlog::warn("Failed to track local map");
            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform track_local_map(). Once the system relocalizes
            // the camera we will use the local map again.
            if (bOK && !visual_odometry)
            {
                bOK = track_local_map();
            }
        }

        if (current_frame.flow_only && !bOK)
        {
            ++flow_recoveries;
            spdlog::debug("Frame {}: extracting features for local-map recovery", current_frame.id);
            extract_current_features();
            bOK = track_with_motion_model();
            if (bOK)
            {
                bOK = track_local_map();
            }
        }

        if (bOK)
        {
            state = OK;
        }
        else if (state == OK)
        {
            if (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd)
            {
                spdlog::debug("Track lost for less than one second...");
                if (!pCurrentMap->is_imu_initialized() || !pCurrentMap->get_inertial_ba2())
                {
                    spdlog::warn("IMU is not or recently initialized; resetting active map");
                    owner_system->reset_active_map();
                }

                state = RECENTLY_LOST;
            }
            else
            {
                state = RECENTLY_LOST; // visual to lost
            }

            lost_timestamp = current_frame.timestamp;
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if ((current_frame.id < (last_relocalization_frame_id + frames_to_reset_imu)) && (current_frame.id > static_cast<unsigned int>(frames_to_reset_imu)) &&
            (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd) && pCurrentMap->is_imu_initialized())
        {
            // TODO check this situation
            spdlog::debug("Saving pointer to frame. imu needs reset...");
            frame *pF          = new frame(current_frame);
            pF->previous_frame = new frame(last_frame);

            // Load preintegration
            pF->frame_imu_preintegrated = new IMU::Preintegrated(current_frame.frame_imu_preintegrated);
        }

        if (pCurrentMap->is_imu_initialized())
        {
            if (bOK)
            {
                if (current_frame.id == (last_relocalization_frame_id + frames_to_reset_imu))
                {
                    spdlog::debug("Resetting frame");
                    reset_frame_imu();
                }
                else if (current_frame.id > (last_relocalization_frame_id + 30))
                {
                    last_bias = current_frame.imu_bias;
                }
            }
        }

        if (bOK || state == RECENTLY_LOST)
        {
            // Update motion model
            if (last_frame.is_set() && current_frame.is_set())
            {
                Sophus::SE3f LastTwc = last_frame.get_pose().inverse();
                velocity             = current_frame.get_pose() * LastTwc;
                has_velocity         = true;
            }
            else
            {
                has_velocity = false;
            }

            // Clean VO matches
            for (int i = 0; i < current_frame.feature_count; i++)
            {
                map_point *pMP = current_frame.map_points[i];
                if (pMP)
                {
                    if (pMP->get_observation_count() < 1)
                    {
                        current_frame.outliers[i]   = false;
                        current_frame.map_points[i] = static_cast<map_point *>(NULL);
                    }
                }
            }

            // Delete temporal MapPoints
            for (list<map_point *>::iterator lit = temporal_map_points.begin(), lend = temporal_map_points.end(); lit != lend; lit++)
            {
                map_point *pMP = *lit;
                delete pMP;
            }
            temporal_map_points.clear();

            bool bNeedKF = need_new_key_frame();

            // Check if we need to insert a new keyframe
            if (bNeedKF && (bOK || (insert_key_frames_when_lost && state == RECENTLY_LOST && (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd))))
            {
                create_new_key_frame();
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for (int i = 0; i < current_frame.feature_count; i++)
            {
                if (current_frame.map_points[i] && current_frame.outliers[i])
                {
                    current_frame.map_points[i] = static_cast<map_point *>(NULL);
                }
            }
        }

        // Reset if the camera get lost soon after initialization
        if (state == LOST)
        {
            if (pCurrentMap->key_frame_count() <= 10)
            {
                owner_system->reset_active_map();
                return;
            }
            if (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd)
            {
                if (!pCurrentMap->is_imu_initialized())
                {
                    spdlog::warn("Track lost before IMU initialisation; resetting");
                    owner_system->reset_active_map();
                    return;
                }
            }

            create_map_in_atlas();

            return;
        }

        if (!current_frame.reference_key_frame)
        {
            current_frame.reference_key_frame = reference_key_frame;
        }

        last_frame = frame(current_frame);
    }

    if (state == OK || state == RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (current_frame.is_set())
        {
            Sophus::SE3f Tcr_ = current_frame.get_pose() * current_frame.reference_key_frame->get_pose_inverse();
            relative_frame_poses.push_back(Tcr_);
            reference_key_frames.push_back(current_frame.reference_key_frame);
            frame_times.push_back(current_frame.timestamp);
            lost_frames.push_back(state == LOST);
        }
        else
        {
            // This can happen if tracking is lost
            relative_frame_poses.push_back(relative_frame_poses.back());
            reference_key_frames.push_back(reference_key_frames.back());
            frame_times.push_back(frame_times.back());
            lost_frames.push_back(state == LOST);
        }
    }

#ifdef REGISTER_LOOP
    if (stop())
    {

        // Safe area to stop
        while (is_stopped())
        {
            usleep(3000);
        }
    }
#endif
}

void tracking::stereo_initialization()
{
    if (current_frame.feature_count > 500)
    {
        if (sensor == system::imu_stereo || sensor == system::imu_rgbd)
        {
            if (!current_frame.imu_preintegrated || !last_frame.imu_preintegrated)
            {
                spdlog::debug("No IMU measurements");
                return;
            }

            if (!fast_initialization && (current_frame.frame_imu_preintegrated->avgA - last_frame.frame_imu_preintegrated->avgA).norm() < 0.5)
            {
                spdlog::debug("Insufficient acceleration for initialization");
                return;
            }

            if (imu_preintegrated_from_last_key_frame)
            {
                delete imu_preintegrated_from_last_key_frame;
            }

            imu_preintegrated_from_last_key_frame = new IMU::Preintegrated(IMU::Bias(), *imu_calibration);
            current_frame.imu_preintegrated       = imu_preintegrated_from_last_key_frame;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (sensor == system::imu_stereo || sensor == system::imu_rgbd)
        {
            Eigen::Matrix3f Rwb0 = current_frame.imu_calibration.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = current_frame.imu_calibration.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            current_frame.set_imu_pose_velocity(Rwb0, twb0, Vwb0);
        }
        else
        {
            current_frame.set_pose(Sophus::SE3f());
        }

        // Create key_frame
        key_frame *pKFini = new key_frame(current_frame, owner_atlas->get_current_map(), database, cameras);

        // Insert key_frame in the map
        owner_atlas->add_key_frame(pKFini);

        // Create MapPoints and asscoiate to key_frame
        if (!is_fisheye_stereo(cameras))
        {
            for (int i = 0; i < current_frame.feature_count; i++)
            {
                float z = current_frame.depths[i];
                if (z > 0)
                {
                    Eigen::Vector3f x3D;
                    current_frame.unproject_stereo(i, x3D);
                    map_point *pNewMP = new map_point(x3D, pKFini, owner_atlas->get_current_map());
                    pNewMP->add_observation(pKFini, i);
                    pKFini->add_map_point(pNewMP, i);
                    pNewMP->compute_distinctive_descriptors();
                    pNewMP->update_normal_and_depth();
                    owner_atlas->add_map_point(pNewMP);

                    current_frame.map_points[i] = pNewMP;
                }
            }
        }
        else
        {
            for (int i = 0; i < static_cast<int>(current_frame.keypoints.size()); i++)
            {
                int rightIndex = current_frame.left_to_right_matches[i];
                if (rightIndex != -1)
                {
                    Eigen::Vector3f x3D = current_frame.stereo_3d_points[i];

                    map_point *pNewMP = new map_point(x3D, pKFini, owner_atlas->get_current_map());

                    pNewMP->add_observation(pKFini, i);
                    pNewMP->add_observation(pKFini, rightIndex + static_cast<int>(current_frame.keypoints.size()));

                    pKFini->add_map_point(pNewMP, i);
                    pKFini->add_map_point(pNewMP, rightIndex + static_cast<int>(current_frame.keypoints.size()));

                    pNewMP->compute_distinctive_descriptors();
                    pNewMP->update_normal_and_depth();
                    owner_atlas->add_map_point(pNewMP);

                    current_frame.map_points[i]                                                             = pNewMP;
                    current_frame.map_points[rightIndex + static_cast<int>(current_frame.keypoints.size())] = pNewMP;
                }
            }
        }

        spdlog::info("New Map created with {} points", owner_atlas->map_points_in_map());

        local_mapper->insert_key_frame(pKFini);

        last_frame        = frame(current_frame);
        last_key_frame_id = current_frame.id;
        last_key_frame    = pKFini;

        local_key_frames.push_back(pKFini);
        local_map_points                  = owner_atlas->get_all_map_points();
        reference_key_frame               = pKFini;
        current_frame.reference_key_frame = pKFini;

        owner_atlas->set_reference_map_points(local_map_points);

        owner_atlas->get_current_map()->key_frame_origins.push_back(pKFini);

        state = OK;
    }
}

void tracking::monocular_initialization()
{

    if (!ready_to_initialize)
    {
        // Set Reference Frame
        if (current_frame.keypoints.size() > 100)
        {

            initial_frame = frame(current_frame);
            last_frame    = frame(current_frame);
            previous_matched_keypoints.resize(current_frame.keypoints.size());
            for (size_t i = 0; i < current_frame.keypoints.size(); i++)
            {
                previous_matched_keypoints[i] = current_frame.keypoints[i].pt_undistorted;
            }

            fill(initial_matches.begin(), initial_matches.end(), -1);

            if (sensor == system::imu_monocular)
            {
                if (imu_preintegrated_from_last_key_frame)
                {
                    delete imu_preintegrated_from_last_key_frame;
                }
                imu_preintegrated_from_last_key_frame = new IMU::Preintegrated(IMU::Bias(), *imu_calibration);
                current_frame.imu_preintegrated       = imu_preintegrated_from_last_key_frame;
            }

            ready_to_initialize = true;

            return;
        }
    }
    else
    {
        if (((int)current_frame.keypoints.size() <= 100) || ((sensor == system::imu_monocular) && (last_frame.timestamp - initial_frame.timestamp > 1.0)))
        {
            ready_to_initialize = false;

            return;
        }

        // Find correspondences
        descriptor_matcher matcher(0.9, true);
        int                nmatches = matcher.search_for_initialization(initial_frame, current_frame, previous_matched_keypoints, initial_matches, 100);

        // Check if there are enough correspondences
        if (nmatches < 100)
        {
            ready_to_initialize = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        const auto initial_keys = to_cv_keypoints(initial_frame.keypoints, true);
        const auto current_keys = to_cv_keypoints(current_frame.keypoints, true);
        if (cameras[0]->reconstruct_with_two_views(initial_keys, current_keys, initial_matches, Tcw, initial_3d_points, vbTriangulated))
        {
            for (size_t i = 0, iend = initial_matches.size(); i < iend; i++)
            {
                if (initial_matches[i] >= 0 && !vbTriangulated[i])
                {
                    initial_matches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            initial_frame.set_pose(Sophus::SE3f());
            current_frame.set_pose(Tcw);

            create_initial_map_monocular();
        }
    }
}

void tracking::create_initial_map_monocular()
{
    // Create KeyFrames
    key_frame *pKFini = new key_frame(initial_frame, owner_atlas->get_current_map(), database, cameras);
    key_frame *pKFcur = new key_frame(current_frame, owner_atlas->get_current_map(), database, cameras);

    if (sensor == system::imu_monocular)
    {
        pKFini->imu_preintegrated = (IMU::Preintegrated *)(NULL);
    }

    pKFini->compute_bow();
    pKFcur->compute_bow();

    // Insert KFs in the map
    owner_atlas->add_key_frame(pKFini);
    owner_atlas->add_key_frame(pKFcur);

    for (size_t i = 0; i < initial_matches.size(); i++)
    {
        if (initial_matches[i] < 0)
        {
            continue;
        }

        // Create map_point.
        Eigen::Vector3f worldPos;
        worldPos << initial_3d_points[i].x, initial_3d_points[i].y, initial_3d_points[i].z;
        map_point *pMP = new map_point(worldPos, pKFcur, owner_atlas->get_current_map());

        pKFini->add_map_point(pMP, i);
        pKFcur->add_map_point(pMP, initial_matches[i]);

        pMP->add_observation(pKFini, i);
        pMP->add_observation(pKFcur, initial_matches[i]);

        pMP->compute_distinctive_descriptors();
        pMP->update_normal_and_depth();

        // Fill Current Frame structure
        current_frame.map_points[initial_matches[i]] = pMP;
        current_frame.outliers[initial_matches[i]]   = false;

        // Add to Map
        owner_atlas->add_map_point(pMP);
    }

    // Update Connections
    pKFini->update_connections();
    pKFcur->update_connections();

    std::set<map_point *> sMPs;
    sMPs = pKFini->get_map_points();

    // Bundle Adjustment
    spdlog::info("New Map created with {} points", owner_atlas->map_points_in_map());
    optimizer::global_bundle_adjustment(owner_atlas->get_current_map(), 20);

    float medianDepth = pKFini->compute_scene_median_depth(2);
    float invMedianDepth;
    if (sensor == system::imu_monocular)
    {
        invMedianDepth = 4.0f / medianDepth; // 4.0f
    }
    else
    {
        invMedianDepth = 1.0f / medianDepth;
    }

    if (medianDepth < 0 || pKFcur->tracked_map_points(1) < 50) // TODO Check, originally 100 tracks
    {
        spdlog::warn("Wrong initialization; resetting");
        owner_system->reset_active_map();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w   = pKFcur->get_pose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->set_pose(Tc2w);

    // Scale points
    vector<map_point *> vpAllMapPoints = pKFini->get_map_point_matches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            map_point *pMP = vpAllMapPoints[iMP];
            pMP->set_world_position(pMP->get_world_position() * invMedianDepth);
            pMP->update_normal_and_depth();
        }
    }

    if (sensor == system::imu_monocular)
    {
        pKFcur->previous_key_frame = pKFini;
        pKFini->next_key_frame     = pKFcur;
        pKFcur->imu_preintegrated  = imu_preintegrated_from_last_key_frame;

        imu_preintegrated_from_last_key_frame = new IMU::Preintegrated(pKFcur->imu_preintegrated->GetUpdatedBias(), pKFcur->imu_calibration);
    }

    local_mapper->insert_key_frame(pKFini);
    local_mapper->insert_key_frame(pKFcur);
    local_mapper->first_timestamp = pKFcur->timestamp;

    current_frame.set_pose(pKFcur->get_pose());
    last_key_frame_id = current_frame.id;
    last_key_frame    = pKFcur;

    local_key_frames.push_back(pKFcur);
    local_key_frames.push_back(pKFini);
    local_map_points                  = owner_atlas->get_all_map_points();
    reference_key_frame               = pKFcur;
    current_frame.reference_key_frame = pKFcur;

    // Compute here initial velocity
    vector<key_frame *> vKFs = owner_atlas->get_all_key_frames();

    Sophus::SE3f deltaT = vKFs.back()->get_pose() * vKFs.front()->get_pose_inverse();
    has_velocity        = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux  = (current_frame.timestamp - last_frame.timestamp) / (current_frame.timestamp - initial_frame.timestamp);
    phi        *= aux;

    last_frame = frame(current_frame);

    owner_atlas->set_reference_map_points(local_map_points);

    owner_atlas->get_current_map()->key_frame_origins.push_back(pKFini);

    state = OK;

    initialization_id = pKFcur->id;
}

void tracking::create_map_in_atlas()
{
    last_initialization_frame_id = current_frame.id;
    owner_atlas->create_new_map();
    if (sensor == system::imu_stereo || sensor == system::imu_monocular || sensor == system::imu_rgbd)
    {
        owner_atlas->set_inertial_sensor();
    }
    initialization_set = false;

    initial_frame_id = current_frame.id + 1;
    state            = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    has_velocity = false;
    spdlog::debug("First frame id in map: {}", last_initialization_frame_id + 1);
    visual_odometry = false; // Init value for know if there are enough MapPoints in the last KF
    if (sensor == system::monocular || sensor == system::imu_monocular)
    {
        ready_to_initialize = false;
    }

    if ((sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd) && imu_preintegrated_from_last_key_frame)
    {
        delete imu_preintegrated_from_last_key_frame;
        imu_preintegrated_from_last_key_frame = new IMU::Preintegrated(IMU::Bias(), *imu_calibration);
    }

    if (last_key_frame)
    {
        last_key_frame = static_cast<key_frame *>(NULL);
    }

    if (reference_key_frame)
    {
        reference_key_frame = static_cast<key_frame *>(NULL);
    }

    last_frame    = frame();
    current_frame = frame();
    initial_matches.clear();

    map_created = true;
}

void tracking::check_replaced_in_last_frame()
{
    for (int i = 0; i < last_frame.feature_count; i++)
    {
        map_point *pMP = last_frame.map_points[i];

        if (pMP)
        {
            map_point *pRep = pMP->get_replaced();
            if (pRep)
            {
                last_frame.map_points[i] = pRep;
            }
        }
    }
}

bool tracking::track_reference_key_frame()
{
    extract_current_features();
    // Compute Bag of Words vector
    current_frame.compute_bow();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    descriptor_matcher  matcher(0.7, true);
    vector<map_point *> vpMapPointMatches;

    int nmatches = matcher.search_by_vocabulary(reference_key_frame, current_frame, vpMapPointMatches);

    if (nmatches < 15)
    {
        spdlog::debug("Reference keyframe tracking found fewer than 15 matches");
        return false;
    }

    current_frame.map_points = vpMapPointMatches;
    current_frame.set_pose(last_frame.get_pose());

    optimizer::pose_optimization(&current_frame, cameras);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < current_frame.feature_count; i++)
    {
        if (current_frame.map_points[i])
        {
            if (current_frame.outliers[i])
            {
                map_point *pMP = current_frame.map_points[i];

                current_frame.map_points[i] = static_cast<map_point *>(NULL);
                current_frame.outliers[i]   = false;
                if (current_frame.is_fisheye_stereo && std::cmp_less(i, current_frame.keypoints.size()))
                {
                    pMP->track_in_view = false;
                }
                else
                {
                    pMP->track_in_view_right = false;
                }
                pMP->track_in_view   = false;
                pMP->last_frame_seen = current_frame.id;
                nmatches--;
            }
            else if (current_frame.map_points[i]->get_observation_count() > 0)
            {
                nmatchesMap++;
            }
        }
    }

    if (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd)
    {
        return true;
    }
    else
    {
        return nmatchesMap >= 10;
    }
}

void tracking::update_last_frame()
{
    // Update pose according to reference keyframe
    key_frame   *pRef = last_frame.reference_key_frame;
    Sophus::SE3f Tlr  = relative_frame_poses.back();
    last_frame.set_pose(Tlr * pRef->get_pose());

    if (last_key_frame_id == last_frame.id || sensor == system::monocular || sensor == system::imu_monocular || !only_tracking)
    {
        return;
    }

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int>> vDepthIdx;
    const int                Nfeat = last_frame.is_fisheye_stereo ? static_cast<int>(last_frame.keypoints.size()) : last_frame.feature_count;
    vDepthIdx.reserve(Nfeat);
    for (int i = 0; i < Nfeat; i++)
    {
        float z = last_frame.depths[i];
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

        map_point *pMP = last_frame.map_points[i];

        if (!pMP)
        {
            bCreateNew = true;
        }
        else if (pMP->get_observation_count() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            Eigen::Vector3f x3D;

            if (!last_frame.is_fisheye_stereo)
            {
                last_frame.unproject_stereo(i, x3D);
            }
            else
            {
                x3D = last_frame.unproject_stereo_fisheye(i);
            }

            map_point *pNewMP        = new map_point(x3D, owner_atlas->get_current_map(), &last_frame, i);
            last_frame.map_points[i] = pNewMP;

            temporal_map_points.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > depth_threshold && nPoints > 100)
        {
            break;
        }
    }
}

bool tracking::track_with_motion_model()
{
    descriptor_matcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    update_last_frame();

    const bool imu_prediction = owner_atlas->is_imu_initialized() && (current_frame.id > last_relocalization_frame_id + frames_to_reset_imu);

    if (imu_prediction)
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        predict_state_imu();
    }
    else
    {
        current_frame.set_pose(velocity * last_frame.get_pose());
    }

    if (current_frame.flow_only)
    {
        const auto predicted_pose = current_frame.get_pose();
        const int  flow_matches   = track_with_optical_flow();
        bool       reliable_flow  = false;
        const bool stereo         = sensor == system::stereo || sensor == system::imu_stereo;

        if (flow_matches >= 50)
        {
            optimizer::pose_optimization(&current_frame, cameras);

            const int minimum_inliers = stereo ? std::clamp(flow_reference_matches / 2, 100, 300) : 50;
            reliable_flow             = current_frame.reliable_flow_inliers(gray_image.size(), minimum_inliers);
        }

        if (reliable_flow)
        {
            spdlog::debug("Frame {}: flow-only tracking, {} observations", current_frame.id, flow_matches);
            return true;
        }

        spdlog::debug("Frame {}: optical flow weak; extracting features for descriptor recovery", current_frame.id);
        ++flow_recoveries;
        extract_current_features();
        current_frame.set_pose(predicted_pose);
    }

    fill(current_frame.map_points.begin(), current_frame.map_points.end(), static_cast<map_point *>(NULL));
    if (imu_prediction)
    {
        return true;
    }

    // Project points seen in previous frame
    int th;

    if (sensor == system::stereo)
    {
        th = 7;
    }
    else
    {
        th = 15;
    }

    // TODO: projecting points should be trackers responsiblity, leaving matcher to match pre-projected points.
    int nmatches = matcher.search_by_projection(current_frame, last_frame, th, sensor == system::monocular || sensor == system::imu_monocular, cameras);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        spdlog::debug("Not enough matches, wider window search!!");
        fill(current_frame.map_points.begin(), current_frame.map_points.end(), static_cast<map_point *>(NULL));

        nmatches = matcher.search_by_projection(current_frame, last_frame, 2 * th, sensor == system::monocular || sensor == system::imu_monocular, cameras);
        spdlog::debug("Matches with wider search: " + to_string(nmatches));
    }

    if (nmatches < 20)
    {
        spdlog::debug("Not enough matches!!");
        if (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // Optimize frame pose with all matches
    optimizer::pose_optimization(&current_frame, cameras);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < current_frame.feature_count; i++)
    {
        if (current_frame.map_points[i])
        {
            if (current_frame.outliers[i])
            {
                map_point *pMP = current_frame.map_points[i];

                current_frame.map_points[i] = static_cast<map_point *>(NULL);
                current_frame.outliers[i]   = false;
                if (current_frame.is_fisheye_stereo && std::cmp_less(i, current_frame.keypoints.size()))
                {
                    pMP->track_in_view = false;
                }
                else
                {
                    pMP->track_in_view_right = false;
                }
                pMP->last_frame_seen = current_frame.id;
                nmatches--;
            }
            else if (current_frame.map_points[i]->get_observation_count() > 0)
            {
                nmatchesMap++;
            }
        }
    }

    if (only_tracking)
    {
        visual_odometry = nmatchesMap < 10;
        return nmatches > 20;
    }

    if (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd)
    {
        return true;
    }
    else
    {
        return nmatchesMap >= 10;
    }
}

bool tracking::can_defer_features(double timestamp) const
{
    // Localization-only mode also needs fresh stereo depth for temporary VO points.
    return use_optical_flow && !only_tracking && state == OK && has_velocity && !frame::initial_computations && frame::next_id > last_relocalization_frame_id + 2 &&
           frame::next_id - last_feature_frame_id < static_cast<unsigned long>(std::max(2, maximum_frames / 4)) && timestamp > last_frame.timestamp && timestamp - last_frame.timestamp < 1.0 &&
           !previous_gray.empty() && previous_gray_frame_id == last_frame.id && previous_gray.size() == gray_image.size() && gray_image.type() == CV_8UC1;
}

void tracking::create_frame(double timestamp, keypoint_extractor &extractor)
{
    use_initial_extractor = &extractor == initial_keypoint_extractor.get();
    const bool inertial   = sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd;
    const bool stereo     = sensor == system::stereo || sensor == system::imu_stereo;
    current_frame         = frame(timestamp,
                          place_vocabulary_ptr,
                          camera_matrix,
                          distortion_coefficients,
                          baseline_fx,
                          depth_threshold,
                          inertial ? &last_frame : nullptr,
                          inertial ? *imu_calibration : IMU::Calib(),
                          stereo && is_fisheye_stereo(cameras) ? left_to_right_pose : Sophus::SE3f());

    extract_features(current_frame, extractor, gray_image, cameras, stereo ? right_image : cv::Mat(), stereo ? right_keypoint_extractor.get() : nullptr);
}

void tracking::extract_features(frame &frame, keypoint_extractor &extractor, const cv::Mat &left, const camera_vector &cameras, const cv::Mat &right, keypoint_extractor *right_extractor)
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

void tracking::extract_current_features()
{
    if (!current_frame.flow_only)
    {
        return;
    }
    const bool stereo    = sensor == system::stereo || sensor == system::imu_stereo;
    auto      &extractor = use_initial_extractor ? *initial_keypoint_extractor : *left_keypoint_extractor;
    extract_features(current_frame, extractor, gray_image, cameras, stereo ? right_image : cv::Mat(), stereo ? right_keypoint_extractor.get() : nullptr);
    last_feature_frame_id = current_frame.id;
    ++feature_frames;
}

void tracking::report_flow_statistics() const
{
    if (use_optical_flow)
    {
        spdlog::info("Optical flow: {} flow-only frames, {} feature-extracted frames, {} recovery attempts", flow_frames, feature_frames, flow_recoveries);
    }
}

int tracking::track_with_optical_flow()
{
    flow_reference_matches = 0;
    if (!use_optical_flow || previous_gray.empty() || gray_image.empty() || previous_gray_frame_id != last_frame.id || previous_gray.size() != gray_image.size() ||
        gray_image.type() != CV_8UC1)
    {
        return 0;
    }
    if (last_frame.is_fisheye_stereo && (previous_right_gray.empty() || right_image.empty() || previous_right_gray.size() != right_image.size() || right_image.type() != CV_8UC1))
    {
        return 0;
    }

    const std::size_t previous_keypoint_count = last_frame.keypoints.size() + (last_frame.is_fisheye_stereo ? last_frame.right_keypoints.size() : 0);
    const std::size_t current_keypoint_count  = current_frame.keypoints.size() + (current_frame.is_fisheye_stereo ? current_frame.right_keypoints.size() : 0);

    if (previous_keypoint_count != static_cast<std::size_t>(last_frame.feature_count) || current_keypoint_count != static_cast<std::size_t>(current_frame.feature_count) ||
        last_frame.feature_count != static_cast<int>(last_frame.map_points.size()) || last_frame.feature_count != static_cast<int>(last_frame.outliers.size()) ||
        current_frame.feature_count != static_cast<int>(current_frame.map_points.size()))
    {
        return 0;
    }

    std::array<std::vector<keypoint>, 2>    tracked_keys;
    std::array<std::vector<map_point *>, 2> tracked_points;
    const int                               view_count = last_frame.is_fisheye_stereo ? 2 : 1;
    for (int view = 0; view < view_count; ++view)
    {
        const bool                      right          = view == 1;
        const auto                     &previous_image = right ? previous_right_gray : previous_gray;
        const auto                     &current_image  = right ? right_image : gray_image;
        const auto                     &keys           = right ? last_frame.right_keypoints : last_frame.keypoints;
        const auto                      offset         = right ? last_frame.keypoints.size() : 0;
        const auto                      camera_pose    = right ? current_frame.get_relative_pose_trl() * current_frame.get_pose() : current_frame.get_pose();
        std::vector<cv::Point2f>        previous_points;
        std::vector<cv::Point2f>        current_points;
        std::vector<keypoint>           previous_keys;
        std::vector<map_point *>        map_points;
        std::array<int, 16>             cell_counts{};
        std::unordered_set<map_point *> selected_points;
        previous_points.reserve(keys.size());
        map_points.reserve(keys.size());

        for (std::size_t index = 0; index < keys.size(); ++index)
        {
            map_point *map_point = last_frame.map_points[offset + index];

            if (!map_point || last_frame.outliers[offset + index] || map_point->is_bad() || map_point->get_observation_count() == 0)
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
            if (is_fisheye_stereo(cameras) || distortion_coefficients.at<float>(0) == 0.0f)
            {
                const Eigen::Vector3f position = camera_pose * map_point->get_world_position();
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
        flow_reference_matches += static_cast<int>(previous_points.size());

        std::vector<unsigned char> forward_status;
        std::vector<float>         forward_error;
        const cv::Size             window_size(21, 21);
        const cv::TermCriteria     criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        const auto                 flow_started     = std::chrono::steady_clock::now();
        const auto                &previous_pyramid = last_frame.get_optical_flow_pyramid(previous_image, right);
        const auto                &current_pyramid  = current_frame.get_optical_flow_pyramid(current_image, right);
        cv::calcOpticalFlowPyrLK(previous_pyramid, current_pyramid, previous_points, current_points, forward_status, forward_error, window_size, 3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

        std::vector<cv::Point2f>   backward_points;
        std::vector<unsigned char> backward_status;
        std::vector<float>         backward_error;
        cv::calcOpticalFlowPyrLK(current_pyramid, previous_pyramid, current_points, backward_points, backward_status, backward_error, window_size, 3, criteria);
        current_frame.optical_flow_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - flow_started).count();

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

    current_frame.flow_candidates = flow_reference_matches;
    current_frame.set_tracked_features(std::move(tracked_keys[0]), std::move(tracked_points[0]), std::move(tracked_keys[1]), std::move(tracked_points[1]));
    current_frame.flow_tracks = current_frame.feature_count;
    return current_frame.feature_count >= 0.6 * flow_reference_matches ? current_frame.feature_count : 0;
}

bool tracking::track_local_map()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    tracked_frame_count++;

    update_local_map();
    search_local_points();

    // TOO check outliers before PO
    int aux1 = 0, aux2 = 0;
    for (int i = 0; i < current_frame.feature_count; i++)
    {
        if (current_frame.map_points[i])
        {
            aux1++;
            if (current_frame.outliers[i])
            {
                aux2++;
            }
        }
    }

    if (!owner_atlas->is_imu_initialized())
    {
        optimizer::pose_optimization(&current_frame, cameras);
    }
    else
    {
        if (current_frame.id <= last_relocalization_frame_id + frames_to_reset_imu)
        {
            spdlog::debug("TLM: PoseOptimization ");
            optimizer::pose_optimization(&current_frame, cameras);
        }
        else
        {
            // A flow recovery can reoptimize this frame after the last-frame prior was consumed.
            delete current_frame.imu_constraint;
            current_frame.imu_constraint = nullptr;
            if (!map_updated && last_frame.imu_constraint)
            {
                spdlog::trace("TLM: PoseInertialOptimizationLastFrame ");
                optimizer::pose_inertial_optimization_last_frame(&current_frame, cameras);
            }
            else
            {
                spdlog::trace("TLM: PoseInertialOptimizationLastKeyFrame ");
                optimizer::pose_inertial_optimization_last_key_frame(&current_frame, cameras);
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for (int i = 0; i < current_frame.feature_count; i++)
    {
        if (current_frame.map_points[i])
        {
            aux1++;
            if (current_frame.outliers[i])
            {
                aux2++;
            }
        }
    }

    matches_inliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < current_frame.feature_count; i++)
    {
        if (current_frame.map_points[i])
        {
            if (!current_frame.outliers[i])
            {
                current_frame.map_points[i]->increase_found();
                if (!only_tracking)
                {
                    if (current_frame.map_points[i]->get_observation_count() > 0)
                    {
                        matches_inliers++;
                    }
                }
                else
                {
                    matches_inliers++;
                }
            }
            else if (sensor == system::stereo)
            {
                current_frame.map_points[i] = static_cast<map_point *>(NULL);
            }
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    local_mapper->inlier_match_count = matches_inliers;
    if (current_frame.flow_only && !current_frame.reliable_flow_inliers(gray_image.size()))
    {
        return false;
    }
    if (current_frame.id < last_relocalization_frame_id + maximum_frames && matches_inliers < 50)
    {
        return false;
    }

    if ((matches_inliers > 10) && (state == RECENTLY_LOST))
    {
        return true;
    }

    if (sensor == system::imu_monocular)
    {
        if ((matches_inliers < 15 && owner_atlas->is_imu_initialized()) || (matches_inliers < 50 && !owner_atlas->is_imu_initialized()))
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    else if (sensor == system::imu_stereo || sensor == system::imu_rgbd)
    {
        if (matches_inliers < 15)
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
        if (matches_inliers < 30)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

bool tracking::need_new_key_frame()
{
    if ((sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd) && !owner_atlas->get_current_map()->is_imu_initialized())
    {
        if (sensor == system::imu_monocular && (current_frame.timestamp - last_key_frame->timestamp) >= 0.25)
        {
            return true;
        }
        else if ((sensor == system::imu_stereo || sensor == system::imu_rgbd) && (current_frame.timestamp - last_key_frame->timestamp) >= 0.25)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    if (only_tracking)
    {
        return false;
    }

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (local_mapper->is_stopped() || local_mapper->stop_requested())
    {
        /*if(mSensor == system::monocular)
        {
            spdlog::debug("Keyframe creation deferred because local mapping is stopped");
        }*/
        return false;
    }

    const int nKFs = owner_atlas->key_frames_in_map();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (current_frame.id < last_relocalization_frame_id + maximum_frames && nKFs > maximum_frames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
    {
        nMinObs = 2;
    }
    int nRefMatches = reference_key_frame->tracked_map_points(nMinObs);
    // Flow tracks a bounded subset, not the full extracted feature population.
    if (current_frame.flow_only)
    {
        nRefMatches = std::min(nRefMatches, flow_reference_matches);
    }

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = local_mapper->accept_key_frames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose    = 0;

    if (sensor != system::monocular && sensor != system::imu_monocular)
    {
        int N = current_frame.is_fisheye_stereo ? static_cast<int>(current_frame.keypoints.size()) : current_frame.feature_count;
        for (int i = 0; i < N; i++)
        {
            if (current_frame.depths[i] > 0 && current_frame.depths[i] < depth_threshold)
            {
                if (current_frame.map_points[i] && !current_frame.outliers[i])
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
    if(nClosedPoints < thStereoClosedPoints && (mSensor==system::stereo || mSensor==system::imu_stereo))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if (sensor == system::monocular)
    {
        thRefRatio = 0.9f;
    }

    if (is_fisheye_stereo(cameras))
    {
        thRefRatio = 0.75f;
    }

    if (sensor == system::imu_monocular)
    {
        if (matches_inliers > 350) // Points tracked from the local map
        {
            thRefRatio = 0.75f;
        }
        else
        {
            thRefRatio = 0.90f;
        }
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = current_frame.id >= last_key_frame_id + maximum_frames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((current_frame.id >= last_key_frame_id + minimum_frames) && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    const bool c1c = sensor != system::monocular && sensor != system::imu_monocular && sensor != system::imu_stereo && sensor != system::imu_rgbd &&
                     (matches_inliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((matches_inliers < nRefMatches * thRefRatio || bNeedToInsertClose)) && matches_inliers > 15);

    //  Temporal condition for Inertial cases
    bool c3 = false;
    if (last_key_frame)
    {
        if (sensor == system::imu_monocular)
        {
            if ((current_frame.timestamp - last_key_frame->timestamp) >= 0.5)
            {
                c3 = true;
            }
        }
        else if (sensor == system::imu_stereo || sensor == system::imu_rgbd)
        {
            if ((current_frame.timestamp - last_key_frame->timestamp) >= 0.5)
            {
                c3 = true;
            }
        }
    }

    bool c4 = false;
    if ((((matches_inliers < 75) && (matches_inliers > 15)) || state == RECENTLY_LOST) && (sensor == system::imu_monocular))
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
        if (bLocalMappingIdle || local_mapper->is_initializing())
        {
            return true;
        }
        else
        {
            local_mapper->interrupt_bundle_adjustment();
            if (sensor != system::monocular && sensor != system::imu_monocular)
            {
                if (local_mapper->key_frames_in_queue() < 3)
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

void tracking::create_new_key_frame()
{
    if (local_mapper->is_initializing() && !owner_atlas->is_imu_initialized())
    {
        return;
    }
    if (current_frame.flow_only)
    {
        const frame flow_frame(current_frame);
        extract_current_features();
        descriptor_matcher matcher(0.9, true);
        matcher.search_by_projection(current_frame, last_frame, sensor == system::stereo ? 7 : 15, sensor == system::monocular || sensor == system::imu_monocular, cameras);
        search_local_points();
        const int inliers = optimizer::pose_optimization(&current_frame, cameras);
        if (inliers < 50)
        {
            spdlog::warn("Frame {}: postponing keyframe; only {} descriptor inliers after feature refresh", current_frame.id, inliers);
            current_frame = flow_frame;
            return;
        }
        matches_inliers = inliers;
        velocity        = current_frame.get_pose() * last_frame.get_pose().inverse();
        if (!need_new_key_frame())
        {
            return;
        }
    }

    if (!local_mapper->set_not_stop(true))
    {
        return;
    }

    key_frame *pKF = new key_frame(current_frame, owner_atlas->get_current_map(), database, cameras);

    if (owner_atlas->is_imu_initialized())
    {
        pKF->has_imu = true;
    }

    pKF->set_new_bias(current_frame.imu_bias);
    reference_key_frame               = pKF;
    current_frame.reference_key_frame = pKF;

    if (last_key_frame)
    {
        pKF->previous_key_frame        = last_key_frame;
        last_key_frame->next_key_frame = pKF;
    }
    else
    {
        spdlog::debug("No last KF in KF creation!!");
    }

    // Reset preintegration from last KF (Create new object)
    if (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd)
    {
        imu_preintegrated_from_last_key_frame = new IMU::Preintegrated(pKF->get_imu_bias(), pKF->imu_calibration);
    }

    if (sensor != system::monocular && sensor != system::imu_monocular) // TODO check if include imu_stereo
    {
        current_frame.update_pose_matrices();
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if (sensor == system::imu_stereo || sensor == system::imu_rgbd)
        {
            maxPoint = 100;
        }

        vector<pair<float, int>> vDepthIdx;
        int                      N = current_frame.is_fisheye_stereo ? static_cast<int>(current_frame.keypoints.size()) : current_frame.feature_count;
        vDepthIdx.reserve(current_frame.feature_count);
        for (int i = 0; i < N; i++)
        {
            float z = current_frame.depths[i];
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

                map_point *pMP = current_frame.map_points[i];
                if (!pMP)
                {
                    bCreateNew = true;
                }
                else if (pMP->get_observation_count() < 1)
                {
                    bCreateNew                  = true;
                    current_frame.map_points[i] = static_cast<map_point *>(NULL);
                }

                if (bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if (!current_frame.is_fisheye_stereo)
                    {
                        current_frame.unproject_stereo(i, x3D);
                    }
                    else
                    {
                        x3D = current_frame.unproject_stereo_fisheye(i);
                    }

                    map_point *pNewMP = new map_point(x3D, pKF, owner_atlas->get_current_map());
                    pNewMP->add_observation(pKF, i);

                    // Check if it is a stereo observation in order to not
                    // duplicate mappoints
                    if (current_frame.is_fisheye_stereo && current_frame.left_to_right_matches[i] >= 0)
                    {
                        current_frame.map_points[static_cast<int>(current_frame.keypoints.size()) + current_frame.left_to_right_matches[i]] = pNewMP;
                        pNewMP->add_observation(pKF, static_cast<int>(current_frame.keypoints.size()) + current_frame.left_to_right_matches[i]);
                        pKF->add_map_point(pNewMP, static_cast<int>(current_frame.keypoints.size()) + current_frame.left_to_right_matches[i]);
                    }

                    pKF->add_map_point(pNewMP, i);
                    pNewMP->compute_distinctive_descriptors();
                    pNewMP->update_normal_and_depth();
                    owner_atlas->add_map_point(pNewMP);

                    current_frame.map_points[i] = pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if (vDepthIdx[j].first > depth_threshold && nPoints > maxPoint)
                {
                    break;
                }
            }
        }
    }

    local_mapper->insert_key_frame(pKF);

    local_mapper->set_not_stop(false);

    last_key_frame_id = current_frame.id;
    last_key_frame    = pKF;
}

void tracking::search_local_points()
{
    // Do not search map points already matched
    for (vector<map_point *>::iterator vit = current_frame.map_points.begin(), vend = current_frame.map_points.end(); vit != vend; vit++)
    {
        map_point *pMP = *vit;
        if (pMP)
        {
            if (pMP->is_bad())
            {
                *vit = static_cast<map_point *>(NULL);
            }
            else
            {
                pMP->increase_visible();
                pMP->last_frame_seen     = current_frame.id;
                pMP->track_in_view       = false;
                pMP->track_in_view_right = false;
            }
        }
    }

    if (current_frame.flow_only)
    {
        return;
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<map_point *>::iterator vit = local_map_points.begin(), vend = local_map_points.end(); vit != vend; vit++)
    {
        map_point *pMP = *vit;

        if (pMP->last_frame_seen == current_frame.id)
        {
            continue;
        }
        if (pMP->is_bad())
        {
            continue;
        }
        // Project (this fills map_point variables for matching)
        if (current_frame.is_in_frustum(pMP, 0.5, cameras))
        {
            pMP->increase_visible();
            nToMatch++;
        }
        if (pMP->track_in_view)
        {
            current_frame.projected_points[pMP->id] = cv::Point2f(pMP->track_projection_x, pMP->track_projection_y);
        }
    }

    if (nToMatch > 0)
    {
        descriptor_matcher matcher(0.8);
        int                th = 1;
        if (sensor == system::rgbd || sensor == system::imu_rgbd)
        {
            th = 3;
        }
        if (owner_atlas->is_imu_initialized())
        {
            if (owner_atlas->get_current_map()->get_inertial_ba2())
            {
                th = 2;
            }
            else
            {
                th = 6;
            }
        }
        else if (!owner_atlas->is_imu_initialized() && (sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd))
        {
            th = 10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if (current_frame.id < last_relocalization_frame_id + 2)
        {
            th = 5;
        }

        if (state == LOST || state == RECENTLY_LOST) // Lost for less than 1 second
        {
            th = 15; // 15
        }

        matcher.search_by_projection(current_frame, local_map_points, th, local_mapper->far_points_enabled, local_mapper->far_point_threshold);
    }
}

void tracking::update_local_map()
{
    // This is for visualization
    owner_atlas->set_reference_map_points(local_map_points);

    // Update
    update_local_key_frames();
    update_local_points();
}

void tracking::update_local_points()
{
    local_map_points.clear();

    int count_pts = 0;

    for (vector<key_frame *>::const_reverse_iterator itKF = local_key_frames.rbegin(), itEndKF = local_key_frames.rend(); itKF != itEndKF; ++itKF)
    {
        key_frame                *pKF   = *itKF;
        const vector<map_point *> vpMPs = pKF->get_map_point_matches();

        for (vector<map_point *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {

            map_point *pMP = *itMP;
            if (!pMP)
            {
                continue;
            }
            if (pMP->track_reference_for_frame == current_frame.id)
            {
                continue;
            }
            if (!pMP->is_bad())
            {
                count_pts++;
                local_map_points.push_back(pMP);
                pMP->track_reference_for_frame = current_frame.id;
            }
        }
    }
}

void tracking::update_local_key_frames()
{
    // Each map point vote for the keyframes in which it has been observed
    std::map<key_frame *, int> keyframeCounter;
    if (!owner_atlas->is_imu_initialized() || (current_frame.id < last_relocalization_frame_id + 2))
    {
        for (int i = 0; i < current_frame.feature_count; i++)
        {
            map_point *pMP = current_frame.map_points[i];
            if (pMP)
            {
                if (!pMP->is_bad())
                {
                    const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();
                    for (std::map<key_frame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    {
                        keyframeCounter[it->first]++;
                    }
                }
                else
                {
                    current_frame.map_points[i] = NULL;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < last_frame.feature_count; i++)
        {
            // Using lastframe since current frame has not matches yet
            if (last_frame.map_points[i])
            {
                map_point *pMP = last_frame.map_points[i];
                if (!pMP)
                {
                    continue;
                }
                if (!pMP->is_bad())
                {
                    const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();
                    for (std::map<key_frame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    {
                        keyframeCounter[it->first]++;
                    }
                }
                else
                {
                    // MODIFICATION
                    last_frame.map_points[i] = NULL;
                }
            }
        }
    }

    int        max    = 0;
    key_frame *pKFmax = static_cast<key_frame *>(NULL);

    local_key_frames.clear();
    local_key_frames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (std::map<key_frame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        key_frame *pKF = it->first;

        if (pKF->is_bad())
        {
            continue;
        }

        if (it->second > max)
        {
            max    = it->second;
            pKFmax = pKF;
        }

        local_key_frames.push_back(pKF);
        pKF->track_reference_for_frame = current_frame.id;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<key_frame *>::const_iterator itKF = local_key_frames.begin(), itEndKF = local_key_frames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (local_key_frames.size() > 80) // 80
        {
            break;
        }

        key_frame *pKF = *itKF;

        const vector<key_frame *> vNeighs = pKF->get_best_covisibility_key_frames(10);

        for (vector<key_frame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            key_frame *pNeighKF = *itNeighKF;
            if (!pNeighKF->is_bad())
            {
                if (pNeighKF->track_reference_for_frame != current_frame.id)
                {
                    local_key_frames.push_back(pNeighKF);
                    pNeighKF->track_reference_for_frame = current_frame.id;
                    break;
                }
            }
        }

        const set<key_frame *> spChilds = pKF->get_children();
        for (set<key_frame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            key_frame *pChildKF = *sit;
            if (!pChildKF->is_bad())
            {
                if (pChildKF->track_reference_for_frame != current_frame.id)
                {
                    local_key_frames.push_back(pChildKF);
                    pChildKF->track_reference_for_frame = current_frame.id;
                    break;
                }
            }
        }

        key_frame *pParent = pKF->get_parent();
        if (pParent)
        {
            if (pParent->track_reference_for_frame != current_frame.id)
            {
                local_key_frames.push_back(pParent);
                pParent->track_reference_for_frame = current_frame.id;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if ((sensor == system::imu_monocular || sensor == system::imu_stereo || sensor == system::imu_rgbd) && local_key_frames.size() < 80)
    {
        key_frame *tempKeyFrame = current_frame.last_key_frame;

        const int Nd = 20;
        for (int i = 0; i < Nd; i++)
        {
            if (!tempKeyFrame)
            {
                break;
            }
            if (tempKeyFrame->track_reference_for_frame != current_frame.id)
            {
                local_key_frames.push_back(tempKeyFrame);
                tempKeyFrame->track_reference_for_frame = current_frame.id;
                tempKeyFrame                            = tempKeyFrame->previous_key_frame;
            }
        }
    }

    if (pKFmax)
    {
        reference_key_frame               = pKFmax;
        current_frame.reference_key_frame = reference_key_frame;
    }
}

bool tracking::relocalization()
{
    extract_current_features();
    spdlog::debug("Starting relocalization");
    // Compute Bag of Words Vector
    current_frame.compute_bow();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query key_frame Database for keyframe candidates for relocalisation
    vector<key_frame *> vpCandidateKFs = database->detect_relocalization_candidates(&current_frame, owner_atlas->get_current_map());

    if (vpCandidateKFs.empty())
    {
        spdlog::debug("There are not candidates");
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    descriptor_matcher matcher(0.75, true);

    vector<vector<map_point *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    for (int i = 0; i < nKFs; i++)
    {
        key_frame *pKF = vpCandidateKFs[i];
        if (pKF->is_bad())
        {
            vbDiscarded[i] = true;
        }
        else
        {
            int nmatches = matcher.search_by_vocabulary(pKF, current_frame, vvpMapPointMatches[i]);
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
            map_point *map_point = vvpMapPointMatches[i][j];
            if (!map_point || map_point->is_bad() || j >= current_frame.keypoints.size())
            {
                continue;
            }
            const cv::KeyPoint   &keypoint = current_frame.keypoints[j].undistorted();
            const Eigen::Vector3f position = map_point->get_world_position();
            world_points.emplace_back(position.x(), position.y(), position.z());
            image_points.push_back(keypoint.pt);
            max_errors.push_back(current_frame.level_sigma_squared[keypoint.octave] * 5.991f);
        }

        vector<bool>    pnp_inliers;
        Eigen::Matrix4f Tcw_matrix;
        const int       min_inliers = std::max(10, static_cast<int>(world_points.size() * 0.5));
        if (!geometric_tools::solve_pnp_ransac(world_points, image_points, max_errors, *cameras[0], min_inliers, Tcw_matrix, pnp_inliers))
        {
            continue;
        }

        current_frame.set_pose(Sophus::SE3f(Tcw_matrix));
        set<map_point *> sFound;
        std::fill(current_frame.map_points.begin(), current_frame.map_points.end(), nullptr);
        size_t correspondence_index = 0;
        for (size_t j = 0; j < vvpMapPointMatches[i].size(); ++j)
        {
            map_point *map_point = vvpMapPointMatches[i][j];
            if (!map_point || map_point->is_bad() || j >= current_frame.keypoints.size())
            {
                continue;
            }
            if (pnp_inliers[correspondence_index])
            {
                current_frame.map_points[j] = map_point;
                sFound.insert(map_point);
            }
            ++correspondence_index;
        }

        int nGood = optimizer::pose_optimization(&current_frame, cameras);

        if (nGood < 10)
        {
            continue;
        }

        for (int io = 0; io < current_frame.feature_count; io++)
        {
            if (current_frame.outliers[io])
            {
                current_frame.map_points[io] = static_cast<map_point *>(NULL);
            }
        }

        // If few inliers, search by projection in a coarse window and optimize again
        if (nGood < 50)
        {
            int nadditional = matcher2.search_by_projection(current_frame, vpCandidateKFs[i], sFound, 10, 100, cameras);

            if (nadditional + nGood >= 50)
            {
                nGood = optimizer::pose_optimization(&current_frame, cameras);

                // If many inliers but still not enough, search by projection again in a narrower window
                // the camera has been already optimized with many points
                if (nGood > 30 && nGood < 50)
                {
                    sFound.clear();
                    for (int ip = 0; ip < current_frame.feature_count; ip++)
                    {
                        if (current_frame.map_points[ip])
                        {
                            sFound.insert(current_frame.map_points[ip]);
                        }
                    }
                    nadditional = matcher2.search_by_projection(current_frame, vpCandidateKFs[i], sFound, 3, 64, cameras);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = optimizer::pose_optimization(&current_frame, cameras);

                        for (int io = 0; io < current_frame.feature_count; io++)
                        {
                            if (current_frame.outliers[io])
                            {
                                current_frame.map_points[io] = NULL;
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
        last_relocalization_frame_id = current_frame.id;
        spdlog::info("Relocalized");
        return true;
    }
}

void tracking::reset(bool bLocMap)
{
    spdlog::debug("System Reseting");
    if (reset_start_callback)
    {
        reset_start_callback();
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        spdlog::debug("Reseting Local Mapper...");
        local_mapper->request_reset();
        spdlog::debug("done");
    }

    // Reset Loop Closing
    spdlog::debug("Reseting Loop Closing...");
    loop_closer->request_reset();
    spdlog::debug("done");

    // Clear BoW Database
    spdlog::debug("Reseting Database...");
    database->clear();
    spdlog::debug("done");

    // Clear Map (this erase MapPoints and KeyFrames)
    owner_atlas->clear_atlas();
    owner_atlas->create_new_map();
    if (sensor == system::imu_stereo || sensor == system::imu_monocular || sensor == system::imu_rgbd)
    {
        owner_atlas->set_inertial_sensor();
    }
    initial_frame_id = 0;

    key_frame::next_id = 0;
    frame::next_id     = 0;
    state              = NO_IMAGES_YET;

    ready_to_initialize = false;
    initialization_set  = false;

    relative_frame_poses.clear();
    reference_key_frames.clear();
    frame_times.clear();
    lost_frames.clear();
    current_frame                = frame();
    last_relocalization_frame_id = 0;
    last_frame                   = frame();
    previous_gray.release();
    previous_right_gray.release();
    reference_key_frame = static_cast<key_frame *>(NULL);
    last_key_frame      = static_cast<key_frame *>(NULL);
    initial_matches.clear();

    if (reset_complete_callback)
    {
        reset_complete_callback();
    }
    spdlog::debug("   End reseting! ");
}

void tracking::reset_active_map(bool bLocMap)
{
    spdlog::debug("Active map Reseting");
    if (reset_start_callback)
    {
        reset_start_callback();
    }
    map *pMap = owner_atlas->get_current_map();

    if (!bLocMap)
    {
        spdlog::debug("Reseting Local Mapper...");
        local_mapper->request_reset_active_map(pMap);
        spdlog::trace("done");
    }

    // Reset Loop Closing
    spdlog::debug("Reseting Loop Closing...");
    loop_closer->request_reset_active_map(pMap);
    spdlog::debug("done");

    // Clear BoW Database
    spdlog::debug("Reseting Database");
    database->clear_map(pMap); // Only clear the active map references
    spdlog::debug("done");

    // Clear Map (this erase MapPoints and KeyFrames)
    owner_atlas->clear_map();

    // key_frame::nNextId = mpAtlas->get_last_init_kf_id();
    // frame::next_id = mnLastInitFrameId;
    last_initialization_frame_id = frame::next_id;
    state                        = NO_IMAGES_YET; // NOT_INITIALIZED;

    ready_to_initialize = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = first_frame_id;
    spdlog::debug("First frame ID: {}", first_frame_id);
    for (map *pMap : owner_atlas->get_all_maps())
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
    spdlog::debug("Initial frame ID: {}", initial_frame_id);

    for (list<bool>::iterator ilbL = lost_frames.begin(); ilbL != lost_frames.end(); ilbL++)
    {
        if (index < initial_frame_id)
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

    lost_frames = lbLost;

    initial_frame_id             = current_frame.id;
    last_relocalization_frame_id = current_frame.id;

    current_frame = frame();
    last_frame    = frame();
    previous_gray.release();
    previous_right_gray.release();
    reference_key_frame = static_cast<key_frame *>(NULL);
    last_key_frame      = static_cast<key_frame *>(NULL);
    initial_matches.clear();

    has_velocity = false;

    if (reset_complete_callback)
    {
        reset_complete_callback();
    }
    spdlog::debug("   End reseting! ");
}

vector<map_point *> tracking::get_local_map_points() { return local_map_points; }

void tracking::inform_only_tracking(const bool &flag) { only_tracking = flag; }

void tracking::update_frame_imu(const float s, const IMU::Bias &b, key_frame *pCurrentKeyFrame)
{
    map                                   *pMap = pCurrentKeyFrame->get_map();
    list<ORB_SLAM3::key_frame *>::iterator lRit = reference_key_frames.begin();
    list<bool>::iterator                   lbL  = lost_frames.begin();
    for (auto lit = relative_frame_poses.begin(), lend = relative_frame_poses.end(); lit != lend; lit++, lRit++, lbL++)
    {
        if (*lbL)
        {
            continue;
        }

        key_frame *pKF = *lRit;

        while (pKF->is_bad())
        {
            pKF = pKF->get_parent();
        }

        if (pKF->get_map() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    last_bias = b;

    last_key_frame = pCurrentKeyFrame;

    last_frame.set_new_bias(last_bias);
    current_frame.set_new_bias(last_bias);

    while (!current_frame.imu_is_preintegrated())
    {
        usleep(500);
    }

    if (last_frame.id == last_frame.last_key_frame->frame_id)
    {
        last_frame.set_imu_pose_velocity(last_frame.last_key_frame->get_imu_rotation(), last_frame.last_key_frame->get_imu_position(), last_frame.last_key_frame->get_velocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = last_frame.last_key_frame->get_imu_position();
        const Eigen::Matrix3f Rwb1 = last_frame.last_key_frame->get_imu_rotation();
        const Eigen::Vector3f Vwb1 = last_frame.last_key_frame->get_velocity();
        float                 t12  = last_frame.imu_preintegrated->dT;

        last_frame.set_imu_pose_velocity(IMU::NormalizeRotation(Rwb1 * last_frame.imu_preintegrated->GetUpdatedDeltaRotation()),
                                         twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * last_frame.imu_preintegrated->GetUpdatedDeltaPosition(),
                                         Vwb1 + Gz * t12 + Rwb1 * last_frame.imu_preintegrated->GetUpdatedDeltaVelocity());
    }

    if (current_frame.imu_preintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = current_frame.last_key_frame->get_imu_position();
        const Eigen::Matrix3f Rwb1 = current_frame.last_key_frame->get_imu_rotation();
        const Eigen::Vector3f Vwb1 = current_frame.last_key_frame->get_velocity();
        float                 t12  = current_frame.imu_preintegrated->dT;

        current_frame.set_imu_pose_velocity(IMU::NormalizeRotation(Rwb1 * current_frame.imu_preintegrated->GetUpdatedDeltaRotation()),
                                            twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * current_frame.imu_preintegrated->GetUpdatedDeltaPosition(),
                                            Vwb1 + Gz * t12 + Rwb1 * current_frame.imu_preintegrated->GetUpdatedDeltaVelocity());
    }

    first_imu_frame_id = current_frame.id;
}

void tracking::new_dataset()
{
    dataset_count++;
    previous_gray.release();
    previous_right_gray.release();
}

int tracking::get_number_dataset() { return dataset_count; }

int tracking::get_matches_inliers() { return matches_inliers; }

void tracking::save_sub_trajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    owner_system->save_trajectory_euroc(strFolder + strNameFile_frames);
    // mpSystem->save_key_frame_trajectory_euroc(strFolder + strNameFile_kf);
}

void tracking::save_sub_trajectory(string strNameFile_frames, string strNameFile_kf, map *pMap)
{
    owner_system->save_trajectory_euroc(strNameFile_frames, pMap);
    if (!strNameFile_kf.empty())
    {
        owner_system->save_key_frame_trajectory_euroc(strNameFile_kf, pMap);
    }
}

float tracking::get_image_scale() { return image_scale; }

#ifdef REGISTER_LOOP
void tracking::request_stop()
{
    unique_lock<mutex> lock(stop_mutex);
    stop_requested_flag = true;
}

bool tracking::stop()
{
    unique_lock<mutex> lock(stop_mutex);
    if (stop_requested_flag && !not_stop)
    {
        stopped = true;
        spdlog::debug("Tracking stopped");
        return true;
    }

    return false;
}

bool tracking::stop_requested()
{
    unique_lock<mutex> lock(stop_mutex);
    return stop_requested_flag;
}

bool tracking::is_stopped()
{
    unique_lock<mutex> lock(stop_mutex);
    return stopped;
}

void tracking::release()
{
    unique_lock<mutex> lock(stop_mutex);
    stopped             = false;
    stop_requested_flag = false;
}
#endif

} // namespace ORB_SLAM3
