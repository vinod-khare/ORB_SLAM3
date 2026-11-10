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

#include "system.h"

#include "local_mapping.h"
#include "loop_closing.h"
#include "settings.h"
#include "tracking.h"
#include "atlas.h"
#include "converter.h"
#include "key_frame_database.h"
#include "place_vocabulary.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

system::system(const string &strVocFile, const string &strSettingsFile, const sensor_type sensor, const int initFr, const string &strSequence, const feature_overrides &overrides)
    : _sensor(sensor), _reset_requested(false), _active_map_reset_requested(false), _localization_mode_activation_requested(false), _localization_mode_deactivation_requested(false), _shutdown(false)
{
    spdlog::info("ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.\n"
                 "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.\n"
                 "This program comes with ABSOLUTELY NO WARRANTY;\n"
                 "This is free software, and you are welcome to redistribute it\n"
                 "under certain conditions. See LICENSE.txt.");

    if (_sensor == monocular)
    {
        spdlog::info("Input sensor: Monocular");
    }
    else if (_sensor == stereo)
    {
        spdlog::info("Input sensor: Stereo");
    }
    else if (_sensor == rgbd)
    {
        spdlog::info("Input sensor: RGB-D");
    }
    else if (_sensor == imu_monocular)
    {
        spdlog::info("Input sensor: Monocular-Inertial");
    }
    else if (_sensor == imu_stereo)
    {
        spdlog::info("Input sensor: Stereo-Inertial");
    }
    else if (_sensor == imu_rgbd)
    {
        spdlog::info("Input sensor: RGB-D-Inertial");
    }

    // Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        spdlog::critical("Failed to open settings file at: {}", strSettingsFile);
        exit(-1);
    }

    _settings = std::make_unique<settings>(strSettingsFile, _sensor, overrides);

    std::ostringstream settings_output;
    settings_output << *_settings;
    spdlog::info("{}", settings_output.str());
    _cameras = std::move(_settings->_cameras);

    cv::FileNode node     = fsSettings["loopClosing"];
    bool         activeLC = true;
    if (!node.empty())
    {
        activeLC = static_cast<int>(fsSettings["loopClosing"]) != 0;
    }

    _vocabulary_file_path = strVocFile;
    _vocabulary           = std::make_unique<place_vocabulary>(strVocFile, _settings->dbow_version());
    _key_frame_database     = std::make_unique<key_frame_database>(*_vocabulary);

    spdlog::info("Initialization of Atlas from scratch");
    _atlas = std::make_unique<atlas>(0);

    if (_sensor == imu_stereo || _sensor == imu_monocular || _sensor == imu_rgbd)
    {
        _atlas->set_inertial_sensor();
    }

    // Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    spdlog::info("Sequence name: {}", strSequence);
    _tracker = std::make_unique<tracking>(this, _vocabulary.get(), _atlas.get(), _key_frame_database.get(), _sensor, _settings.get(), _cameras, strSequence);

    // Initialize the Local Mapping thread and launch
    _local_mapper = std::make_unique<local_mapping>(
        _statistics, this, _atlas.get(), _sensor == monocular || _sensor == imu_monocular, _sensor == imu_monocular || _sensor == imu_stereo || _sensor == imu_rgbd, strSequence);
    _local_mapper->initial_frame      = initFr;
    _local_mapper->far_point_threshold = _settings->far_point_threshold();
    if (_local_mapper->far_point_threshold != 0)
    {
        spdlog::info("Discarding points farther than {} m from the current camera", _local_mapper->far_point_threshold);
        _local_mapper->far_points_enabled = true;
    }
    else
    {
        _local_mapper->far_points_enabled = false;
    }

    // Initialize the Loop Closing thread and launch
    _loop_closer = std::make_unique<loop_closing>(_statistics, _atlas.get(), _key_frame_database.get(), _vocabulary.get(), _sensor != monocular, activeLC);

    // Set pointers between threads
    _tracker->set_local_mapper(_local_mapper.get());
    _tracker->set_loop_closing(_loop_closer.get());

    _local_mapper->set_tracker(_tracker.get());
    _local_mapper->set_loop_closer(_loop_closer.get());

    _loop_closer->set_tracker(_tracker.get());
    _loop_closer->set_local_mapper(_local_mapper.get());

    try
    {
        _local_mapping_thread = std::jthread(&local_mapping::run, _local_mapper.get());
        _loop_closing_thread  = std::jthread(&loop_closing::run, _loop_closer.get());
    }
    catch (...)
    {
        stop_workers();
        throw;
    }
}

system::~system() { stop_workers(); }

int  system::dbow_version() const { return _vocabulary->version(); }
void system::set_dbow_version(int version) { _vocabulary->set_version(version); }

void system::record_frame_statistics(double preprocessing_ms, double frontend_ms)
{
    const auto                         &frame = _tracker->current_frame;
    processing_statistics::frame_sample sample{.frame_id           = frame.id,
                                               .timestamp          = frame.timestamp,
                                               .detected_keypoints = frame.flow_only ? 0 : frame.keypoints.size() + frame.right_keypoints.size(),
                                               .preprocessing_ms   = preprocessing_ms,
                                               .frontend_ms        = frontend_ms};
    sample.extraction_ms       = frame.extraction_ms;
    sample.stereo_matching_ms  = frame.stereo_matching_ms;
    sample.optical_flow_ms     = frame.optical_flow_ms;
    sample.left_keypoints      = frame.keypoints.size();
    sample.right_keypoints     = frame.right_keypoints.size();
    sample.flow_candidates     = frame.flow_candidates;
    sample.flow_tracks         = frame.flow_tracks;
    sample.flow_inliers        = frame.flow_inliers;
    sample.flow_occupied_cells = frame.flow_occupied_cells;
    sample.flow_only           = frame.flow_only;
    sample.features_extracted  = !frame.flow_only;
    sample.tracking_state      = _tracker->state;

    if (auto *map = _atlas->get_current_map())
    {
        sample.map_id           = map->get_id();
        sample.map_change_index = map->get_last_big_change_index();
    }

    if (_tracker->state == tracking::OK)
    {
        for (std::size_t index = 0; index < frame.map_points.size(); ++index)
        {
            if (frame.map_points[index] && index < frame.outliers.size() && !frame.outliers[index])
            {
                ++sample.matched_keypoints;
            }
        }
    }

    _statistics.record_frame(sample);
}

void system::register_frame_filename_for_export(const string &filename)
{
    const size_t pose_count = _tracker->relative_frame_poses.size();

    // Handle tracker resets/restarts: trim stale filenames.
    while (_frame_filenames.size() > pose_count)
    {
        _frame_filenames.pop_back();
    }

    // Handle newly appended frame poses: mirror growth with current filename.
    while (_frame_filenames.size() < pose_count)
    {
        _frame_filenames.push_back(filename);
    }
}

Sophus::SE3f system::track_stereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point> &vImuMeas, string filename)
{
    if (_sensor != stereo && _sensor != imu_stereo)
    {
        spdlog::error("track_stereo called, but the input sensor is not Stereo or Stereo-Inertial");
        exit(-1);
    }

    cv::Mat imLeftToFeed, imRightToFeed;
    if (_settings->need_to_rectify())
    {
        cv::Mat M1l = _settings->left_rectification_map_x();
        cv::Mat M2l = _settings->left_rectification_map_y();
        cv::Mat M1r = _settings->right_rectification_map_x();
        cv::Mat M2r = _settings->right_rectification_map_y();

        cv::remap(imLeft, imLeftToFeed, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightToFeed, M1r, M2r, cv::INTER_LINEAR);
    }
    else if (_settings->need_to_resize())
    {
        cv::resize(imLeft, imLeftToFeed, _settings->new_image_size());
        cv::resize(imRight, imRightToFeed, _settings->new_image_size());
    }
    else
    {
        imLeftToFeed  = imLeft.clone();
        imRightToFeed = imRight.clone();
    }

    // Check mode change
    {
        unique_lock<mutex> lock(_mode_mutex);
        if (_localization_mode_activation_requested)
        {
            _local_mapper->request_stop();

            // Wait until Local Mapping has effectively stopped
            while (!_local_mapper->is_stopped())
            {
                usleep(1000);
            }

            _tracker->inform_only_tracking(true);
            _localization_mode_activation_requested = false;
        }
        if (_localization_mode_deactivation_requested)
        {
            _tracker->inform_only_tracking(false);
            _local_mapper->release();
            _localization_mode_deactivation_requested = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(_reset_mutex);
        if (_reset_requested)
        {
            _tracker->reset();
            _reset_requested          = false;
            _active_map_reset_requested = false;
        }
        else if (_active_map_reset_requested)
        {
            _tracker->reset_active_map();
            _active_map_reset_requested = false;
        }
    }

    if (_sensor == system::imu_stereo)
    {
        for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
        {
            _tracker->grab_imu_data(vImuMeas[i_imu]);
        }
    }

    Sophus::SE3f Tcw = _tracker->grab_image_stereo(imLeftToFeed, imRightToFeed, timestamp, filename);
    register_frame_filename_for_export(filename);

    unique_lock<mutex> lock2(_state_mutex);
    _tracking_state      = _tracker->state;
    _tracked_map_points   = _tracker->current_frame.map_points;
    _tracked_keypoints_undistorted = _tracker->current_frame.keypoints;

    return Tcw;
}

Sophus::SE3f system::track_rgbd(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point> &vImuMeas, string filename)
{
    if (_sensor != rgbd && _sensor != imu_rgbd)
    {
        spdlog::error("track_rgbd called, but the input sensor is not rgbd");
        exit(-1);
    }

    cv::Mat imToFeed      = im.clone();
    cv::Mat imDepthToFeed = depthmap.clone();
    if (_settings->need_to_resize())
    {
        cv::Mat resizedIm;
        cv::resize(im, resizedIm, _settings->new_image_size());
        imToFeed = resizedIm;

        cv::resize(depthmap, imDepthToFeed, _settings->new_image_size());
    }

    // Check mode change
    {
        unique_lock<mutex> lock(_mode_mutex);
        if (_localization_mode_activation_requested)
        {
            _local_mapper->request_stop();

            // Wait until Local Mapping has effectively stopped
            while (!_local_mapper->is_stopped())
            {
                usleep(1000);
            }

            _tracker->inform_only_tracking(true);
            _localization_mode_activation_requested = false;
        }
        if (_localization_mode_deactivation_requested)
        {
            _tracker->inform_only_tracking(false);
            _local_mapper->release();
            _localization_mode_deactivation_requested = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(_reset_mutex);
        if (_reset_requested)
        {
            _tracker->reset();
            _reset_requested          = false;
            _active_map_reset_requested = false;
        }
        else if (_active_map_reset_requested)
        {
            _tracker->reset_active_map();
            _active_map_reset_requested = false;
        }
    }

    if (_sensor == system::imu_rgbd)
    {
        for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
        {
            _tracker->grab_imu_data(vImuMeas[i_imu]);
        }
    }

    Sophus::SE3f Tcw = _tracker->grab_image_rgbd(imToFeed, imDepthToFeed, timestamp, filename);
    register_frame_filename_for_export(filename);

    unique_lock<mutex> lock2(_state_mutex);
    _tracking_state      = _tracker->state;
    _tracked_map_points   = _tracker->current_frame.map_points;
    _tracked_keypoints_undistorted = _tracker->current_frame.keypoints;
    return Tcw;
}

void system::set_next_frame_color(const cv::Mat &color_image, const cv::Mat &color_image_right)
{
    const auto prepare_color = [this](const cv::Mat &image, bool right)
    {
        cv::Mat prepared;
        if (image.empty())
        {
            return prepared;
        }
        if ((_sensor == stereo || _sensor == imu_stereo) && _settings->need_to_rectify())
        {
            cv::remap(image, prepared, right ? _settings->right_rectification_map_x() : _settings->left_rectification_map_x(), right ? _settings->right_rectification_map_y() : _settings->left_rectification_map_y(), cv::INTER_LINEAR);
        }
        else if (_settings->need_to_resize())
        {
            cv::resize(image, prepared, _settings->new_image_size());
        }
        else
        {
            prepared = image.clone();
        }
        return prepared;
    };
    _tracker->next_frame_color        = prepare_color(color_image, false);
    _tracker->next_frame_color_right = prepare_color(color_image_right, true);
}

bool system::can_precompute_optical_flow_pyramid() const
{
    const bool is_stereo = _sensor == stereo || _sensor == imu_stereo;
    const bool is_mono   = _sensor == monocular || _sensor == imu_monocular;
    return _settings->use_optical_flow() && (is_stereo || is_mono) && !_settings->need_to_resize() && (!is_stereo || !_settings->need_to_rectify());
}

void system::set_next_optical_flow_pyramid(std::vector<cv::Mat> pyramid)
{
    if (!can_precompute_optical_flow_pyramid())
    {
        pyramid.clear();
    }
    _tracker->next_optical_flow_pyramid = std::move(pyramid);
}

Sophus::SE3f system::track_monocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point> &vImuMeas, string filename)
{

    {
        unique_lock<mutex> lock(_reset_mutex);
        if (_shutdown)
        {
            return Sophus::SE3f();
        }
    }

    if (_sensor != monocular && _sensor != imu_monocular)
    {
        spdlog::error("track_monocular called, but the input sensor is not Monocular or Monocular-Inertial");
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    if (_settings->need_to_resize())
    {
        cv::Mat resizedIm;
        cv::resize(im, resizedIm, _settings->new_image_size());
        imToFeed = resizedIm;
    }

    // Check mode change
    {
        unique_lock<mutex> lock(_mode_mutex);
        if (_localization_mode_activation_requested)
        {
            _local_mapper->request_stop();

            // Wait until Local Mapping has effectively stopped
            while (!_local_mapper->is_stopped())
            {
                usleep(1000);
            }

            _tracker->inform_only_tracking(true);
            _localization_mode_activation_requested = false;
        }
        if (_localization_mode_deactivation_requested)
        {
            _tracker->inform_only_tracking(false);
            _local_mapper->release();
            _localization_mode_deactivation_requested = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(_reset_mutex);
        if (_reset_requested)
        {
            _tracker->reset();
            _reset_requested          = false;
            _active_map_reset_requested = false;
        }
        else if (_active_map_reset_requested)
        {
            spdlog::info("Resetting active map in monocular case");
            _tracker->reset_active_map();
            _active_map_reset_requested = false;
        }
    }

    if (_sensor == system::imu_monocular)
    {
        for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
        {
            _tracker->grab_imu_data(vImuMeas[i_imu]);
        }
    }

    Sophus::SE3f Tcw = _tracker->grab_image_monocular(imToFeed, timestamp, filename);
    register_frame_filename_for_export(filename);

    unique_lock<mutex> lock2(_state_mutex);
    _tracking_state      = _tracker->state;
    _tracked_map_points   = _tracker->current_frame.map_points;
    _tracked_keypoints_undistorted = _tracker->current_frame.keypoints;

    return Tcw;
}

void system::activate_localization_mode()
{
    unique_lock<mutex> lock(_mode_mutex);
    _localization_mode_activation_requested = true;
}

void system::deactivate_localization_mode()
{
    unique_lock<mutex> lock(_mode_mutex);
    _localization_mode_deactivation_requested = true;
}

bool system::map_changed()
{
    static int n    = 0;
    int        curn = _atlas->get_last_big_change_index();
    if (n < curn)
    {
        n = curn;
        return true;
    }
    else
    {
        return false;
    }
}

void system::reset()
{
    unique_lock<mutex> lock(_reset_mutex);
    _reset_requested = true;
}

void system::reset_active_map()
{
    unique_lock<mutex> lock(_reset_mutex);
    _active_map_reset_requested = true;
}

void system::stop_workers()
{
    _loop_closer->request_finish();
    if (_loop_closing_thread.joinable())
    {
        _loop_closing_thread.join();
    }
    _loop_closer->stop_global_bundle_adjustment();
    _local_mapper->request_finish();
    if (_local_mapping_thread.joinable())
    {
        _local_mapping_thread.join();
    }
}

void system::shutdown()
{
    const std::lock_guard shutdown_lock(_shutdown_mutex);
    {
        unique_lock<mutex> lock(_reset_mutex);
        if (_shutdown)
        {
            return;
        }
        _shutdown = true;
    }

    spdlog::info("shutdown");
    _tracker->report_flow_statistics();

    stop_workers();
}

bool system::is_shut_down()
{
    unique_lock<mutex> lock(_reset_mutex);
    return _shutdown;
}

void system::save_trajectory_tum(const string &filename)
{
    spdlog::info("Saving camera trajectory to {}", filename);
    if (_sensor == monocular)
    {
        spdlog::error("save_trajectory_tum cannot be used for monocular");
        return;
    }

    vector<key_frame *> vpKFs = _atlas->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Two = vpKFs[0]->get_pose_inverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::key_frame *>::iterator lRit = _tracker->reference_key_frames.begin();
    list<double>::iterator                lT   = _tracker->frame_times.begin();
    list<bool>::iterator                  lbL  = _tracker->lost_frames.begin();
    for (list<Sophus::SE3f>::iterator lit = _tracker->relative_frame_poses.begin(), lend = _tracker->relative_frame_poses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
    {
        if (*lbL)
        {
            continue;
        }

        key_frame *pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while (pKF->is_bad())
        {
            Trw = Trw * pKF->pose_relative_to_parent;
            pKF = pKF->get_parent();
        }

        Trw = Trw * pKF->get_pose() * Two;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();

        Eigen::Vector3f    twc = Twc.translation();
        Eigen::Quaternionf q   = Twc.unit_quaternion();

        f << setprecision(6) << *lT << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    f.close();
}

void system::save_key_frame_trajectory_tum(const string &filename)
{
    spdlog::info("Saving keyframe trajectory to {}", filename);

    vector<key_frame *> vpKFs = _atlas->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        key_frame *pKF = vpKFs[i];

        if (pKF->is_bad())
        {
            continue;
        }

        Sophus::SE3f       Twc = pKF->get_pose_inverse();
        Eigen::Quaternionf q   = Twc.unit_quaternion();
        Eigen::Vector3f    t   = Twc.translation();
        f << setprecision(6) << pKF->timestamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }

    f.close();
}

void system::save_trajectory_euroc(const string &filename)
{

    spdlog::info("Saving trajectory to {}", filename);

    vector<map *> vpMaps     = _atlas->get_all_maps();
    size_t        numMaxKFs  = 0;
    map          *pBiggerMap = nullptr;
    spdlog::debug("There are {} maps in the atlas", vpMaps.size());
    for (map *pMap : vpMaps)
    {
        spdlog::debug("Map {} has {} keyframes", pMap->get_id(), pMap->get_all_key_frames().size());
        if (pMap->get_all_key_frames().size() > numMaxKFs)
        {
            numMaxKFs  = pMap->get_all_key_frames().size();
            pBiggerMap = pMap;
        }
    }

    if (!pBiggerMap)
    {
        spdlog::warn("There is no map to save");
        return;
    }

    vector<key_frame *> vpKFs = pBiggerMap->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b depending on IMU or not.
    if (_sensor == imu_monocular || _sensor == imu_stereo || _sensor == imu_rgbd)
    {
        Twb = vpKFs[0]->get_imu_pose();
    }
    else
    {
        Twb = vpKFs[0]->get_pose_inverse();
    }

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::key_frame *>::iterator lRit = _tracker->reference_key_frames.begin();
    list<double>::iterator                lT   = _tracker->frame_times.begin();
    list<bool>::iterator                  lbL  = _tracker->lost_frames.begin();

    for (auto lit = _tracker->relative_frame_poses.begin(), lend = _tracker->relative_frame_poses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
    {
        if (*lbL)
        {
            continue;
        }

        key_frame *pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
        {
            continue;
        }

        while (pKF->is_bad())
        {
            Trw = Trw * pKF->pose_relative_to_parent;
            pKF = pKF->get_parent();
        }

        if (!pKF || pKF->get_map() != pBiggerMap)
        {
            continue;
        }

        Trw = Trw * pKF->get_pose() * Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        if (_sensor == imu_monocular || _sensor == imu_stereo || _sensor == imu_rgbd)
        {
            Sophus::SE3f       Twb = (pKF->imu_calibration.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q   = Twb.unit_quaternion();
            Eigen::Vector3f    twb = Twb.translation();
            f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f       Twc = ((*lit) * Trw).inverse();
            Eigen::Quaternionf q   = Twc.unit_quaternion();
            Eigen::Vector3f    twc = Twc.translation();
            f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
    spdlog::info("Finished saving trajectory to {}", filename);
}

void system::save_trajectory_euroc(const string &filename, map *pMap)
{

    spdlog::info("Saving trajectory of map {} to {}", pMap->get_id(), filename);

    vector<key_frame *> vpKFs = pMap->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (_sensor == imu_monocular || _sensor == imu_stereo || _sensor == imu_rgbd)
    {
        Twb = vpKFs[0]->get_imu_pose();
    }
    else
    {
        Twb = vpKFs[0]->get_pose_inverse();
    }

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::key_frame *>::iterator lRit = _tracker->reference_key_frames.begin();
    list<double>::iterator                lT   = _tracker->frame_times.begin();
    list<bool>::iterator                  lbL  = _tracker->lost_frames.begin();

    for (auto lit = _tracker->relative_frame_poses.begin(), lend = _tracker->relative_frame_poses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
    {
        if (*lbL)
        {
            continue;
        }

        key_frame *pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
        {
            continue;
        }

        while (pKF->is_bad())
        {
            Trw = Trw * pKF->pose_relative_to_parent;
            pKF = pKF->get_parent();
        }

        if (!pKF || pKF->get_map() != pMap)
        {
            continue;
        }

        Trw = Trw * pKF->get_pose() * Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        if (_sensor == imu_monocular || _sensor == imu_stereo || _sensor == imu_rgbd)
        {
            Sophus::SE3f       Twb = (pKF->imu_calibration.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q   = Twb.unit_quaternion();
            Eigen::Vector3f    twb = Twb.translation();
            f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f       Twc = ((*lit) * Trw).inverse();
            Eigen::Quaternionf q   = Twc.unit_quaternion();
            Eigen::Vector3f    twc = Twc.translation();
            f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
    spdlog::info("Finished saving trajectory to {}", filename);
}

void system::save_key_frame_trajectory_euroc(const string &filename)
{
    spdlog::info("Saving keyframe trajectory to {}", filename);

    vector<map *> vpMaps     = _atlas->get_all_maps();
    map          *pBiggerMap = nullptr;
    size_t        numMaxKFs  = 0;
    for (map *pMap : vpMaps)
    {
        if (pMap && pMap->get_all_key_frames().size() > numMaxKFs)
        {
            numMaxKFs  = pMap->get_all_key_frames().size();
            pBiggerMap = pMap;
        }
    }

    if (!pBiggerMap)
    {
        spdlog::warn("There is no map to save");
        return;
    }

    vector<key_frame *> vpKFs = pBiggerMap->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        key_frame *pKF = vpKFs[i];

        if (!pKF || pKF->is_bad())
        {
            continue;
        }
        if (_sensor == imu_monocular || _sensor == imu_stereo || _sensor == imu_rgbd)
        {
            Sophus::SE3f       Twb = pKF->get_imu_pose();
            Eigen::Quaternionf q   = Twb.unit_quaternion();
            Eigen::Vector3f    twb = Twb.translation();
            f << setprecision(6) << 1e9 * pKF->timestamp << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
              << endl;
        }
        else
        {
            Sophus::SE3f       Twc = pKF->get_pose_inverse();
            Eigen::Quaternionf q   = Twc.unit_quaternion();
            Eigen::Vector3f    t   = Twc.translation();
            f << setprecision(6) << 1e9 * pKF->timestamp << " " << setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
              << endl;
        }
    }
    f.close();
}

void system::save_key_frame_trajectory_euroc(const string &filename, map *pMap)
{
    spdlog::info("Saving keyframe trajectory of map {} to {}", pMap->get_id(), filename);

    vector<key_frame *> vpKFs = pMap->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        key_frame *pKF = vpKFs[i];

        if (!pKF || pKF->is_bad())
        {
            continue;
        }
        if (_sensor == imu_monocular || _sensor == imu_stereo || _sensor == imu_rgbd)
        {
            Sophus::SE3f       Twb = pKF->get_imu_pose();
            Eigen::Quaternionf q   = Twb.unit_quaternion();
            Eigen::Vector3f    twb = Twb.translation();
            f << setprecision(6) << 1e9 * pKF->timestamp << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
              << endl;
        }
        else
        {
            Sophus::SE3f       Twc = pKF->get_pose_inverse();
            Eigen::Quaternionf q   = Twc.unit_quaternion();
            Eigen::Vector3f    t   = Twc.translation();
            f << setprecision(6) << 1e9 * pKF->timestamp << " " << setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
              << endl;
        }
    }
    f.close();
}

void system::save_colmap(const string &strDir)
{
    spdlog::debug("");
    spdlog::debug("✨ [save_colmap] Exporting COLMAP-compatible sparse model");
    spdlog::debug("📁 [save_colmap] Output directory: " + strDir);

    // Locate the largest map (same logic as save_trajectory_euroc)
    vector<map *> vpMaps     = _atlas->get_all_maps();
    map          *pBiggerMap = nullptr;
    int           numMaxKFs  = 0;
    for (map *pMap : vpMaps)
    {
        if ((int)pMap->get_all_key_frames().size() > numMaxKFs)
        {
            numMaxKFs  = pMap->get_all_key_frames().size();
            pBiggerMap = pMap;
        }
    }
    if (!pBiggerMap)
    {
        spdlog::debug("❌ [save_colmap] No map found. Nothing to export.");
        return;
    }

    std::filesystem::create_directories(strDir);

    vector<key_frame *> vpKFs = pBiggerMap->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);
    // World origin anchored at the first KF (monocular)
    Sophus::SE3f Twb = vpKFs[0]->get_pose_inverse();

    // ------------------------------------------------------------------
    // cameras.txt
    // COLMAP model: PINHOLE -> fx fy cx cy
    //               OPENCV_FISHEYE -> fx fy cx cy k1 k2 k3 k4
    // ------------------------------------------------------------------
    {
        ofstream fc(strDir + "/cameras.txt");
        fc << fixed << setprecision(9);
        fc << "# Camera list with one line of data per camera:\n";
        fc << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
        fc << "# Number of cameras: 1\n";

        int         width  = _settings->new_image_size().width;
        int         height = _settings->new_image_size().height;
        const auto &cam    = _cameras[0];

        if (cam)
        {
            if (cam->get_type() == geometric_camera::cam_pinhole)
            {
                // parameters: [fx, fy, cx, cy]
                fc << "1 PINHOLE " << width << " " << height << " " << cam->get_parameter(0) << " " << cam->get_parameter(1) << " " << cam->get_parameter(2) << " " << cam->get_parameter(3)
                   << "\n";
            }
            else
            {
                // KannalaBrandt8: [fx, fy, cx, cy, k1, k2, k3, k4]
                fc << "1 OPENCV_FISHEYE " << width << " " << height << " " << cam->get_parameter(0) << " " << cam->get_parameter(1) << " " << cam->get_parameter(2) << " " << cam->get_parameter(3)
                   << " " << cam->get_parameter(4) << " " << cam->get_parameter(5) << " " << cam->get_parameter(6) << " " << cam->get_parameter(7) << "\n";
            }
        }
        else
        {
            fc << "# WARNING: camera calibration not available\n";
        }
        fc.close();
    }

    // ------------------------------------------------------------------
    // points3D.txt  (empty - filled by COLMAP point_triangulator)
    // ------------------------------------------------------------------
    {
        ofstream fp(strDir + "/points3D.txt");
        fp << "# 3D point list with one line of data per point:\n";
        fp << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
        fp << "# Number of points: 0\n";
        fp.close();
    }

    // ------------------------------------------------------------------
    // images.txt + poses_debug.csv
    //
    // ORB-SLAM3 native format: Twc (camera-to-world)
    //   Twc.translation() = camera position in world frame
    //   Twc.unit_quaternion() = rotation R_wc (world <- camera)
    //
    // COLMAP images.txt expects Tcw (world-to-camera):
    //   LINE: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    //   where Tcw = Twc.inverse()
    //   quaternion order: qw qx qy qz
    //   translation (TX,TY,TZ) = Tcw.translation() = -R_cw * t_wc
    // ------------------------------------------------------------------
    ofstream fi(strDir + "/images.txt");
    fi << fixed << setprecision(9);
    fi << "# Image list with two lines of data per image:\n";
    fi << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fi << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fi << "# Pose convention: world-to-camera (Tcw). Quaternion order: qw qx qy qz.\n";
    fi << "# To recover camera position in world frame: t_wc = -R_cw^T * Tcw.translation()\n";

    ofstream fcsv(strDir + "/poses_debug.csv");
    fcsv << fixed << setprecision(9);
    // CSV stores Twc (camera-to-world) for easy inspection
    fcsv << "# ORB-SLAM3 per-frame pose export (camera-to-world / Twc convention)\n";
    fcsv << "# tx ty tz = camera position in world frame\n";
    fcsv << "# qx qy qz qw = quaternion of R_wc (world <- camera), Eigen/ROS convention (w last)\n";
    fcsv << "# COLMAP conversion: Tcw = Twc.inverse(); use qw qx qy qz and Tcw.translation()\n";
    fcsv << "image_name,timestamp,tx,ty,tz,qx,qy,qz,qw,pose_convention\n";

    list<ORB_SLAM3::key_frame *>::iterator lRit = _tracker->reference_key_frames.begin();
    list<double>::iterator                lT   = _tracker->frame_times.begin();
    list<bool>::iterator                  lbL  = _tracker->lost_frames.begin();
    list<string>::iterator                lF   = _frame_filenames.begin();

    int          image_id = 1;
    int          dropped  = 0;
    int          exported = 0;
    Sophus::SE3f last_Tcw = Sophus::SE3f(); // Identity for first frame if needed

    for (auto lit = _tracker->relative_frame_poses.begin(), lend = _tracker->relative_frame_poses.end(); lit != lend; ++lit, ++lRit, ++lT, ++lbL)
    {
        // Always try to export this frame
        Sophus::SE3f Tcw;
        bool         has_pose = false;

        if (!*lbL && *lRit)
        {
            key_frame    *pKF = *lRit;
            Sophus::SE3f Trw;
            while (pKF->is_bad())
            {
                Trw = Trw * pKF->pose_relative_to_parent;
                pKF = pKF->get_parent();
            }
            if (pKF && pKF->get_map() == pBiggerMap)
            {
                Trw      = Trw * pKF->get_pose() * Twb;
                Tcw      = (*lit) * Trw; // world-to-camera
                last_Tcw = Tcw;
                has_pose = true;
            }
        }

        if (!has_pose)
        {
            // Use last known pose for dropped frames
            Tcw = last_Tcw;
            std::ostringstream oss;
            oss << "⚠️  [save_colmap] Using last pose for dropped frame, ts=" << std::fixed << std::setprecision(6) << *lT;
            spdlog::trace(oss.str());
            ++dropped;
        }

        Sophus::SE3f Twc = Tcw.inverse(); // camera-to-world

        // Recover image filename from stored path (basename only)
        string img_name;
        if (lF != _frame_filenames.end() && !lF->empty())
        {
            img_name = std::filesystem::path(*lF).filename().string();
        }
        else
        {
            // Fallback: reconstruct TUM-VI style name from timestamp (nanoseconds)
            long long ns = llround(*lT * 1e9);
            img_name     = to_string(ns) + ".png";
        }
        if (lF != _frame_filenames.end())
        {
            ++lF;
        }

        // --- COLMAP images.txt row (world-to-camera, qw qx qy qz) ---
        Eigen::Quaternionf q_cw = Tcw.unit_quaternion();
        Eigen::Vector3f    t_cw = Tcw.translation();
        fi << image_id << " " << q_cw.w() << " " << q_cw.x() << " " << q_cw.y() << " " << q_cw.z() << " " << t_cw(0) << " " << t_cw(1) << " " << t_cw(2) << " "
           << "1 " << img_name << "\n";
        fi << "\n"; // empty 2D-point observations line

        // --- debug CSV row (camera-to-world / Twc) ---
        Eigen::Quaternionf q_wc = Twc.unit_quaternion();
        Eigen::Vector3f    t_wc = Twc.translation();
        fcsv << img_name << "," << *lT << "," << t_wc(0) << "," << t_wc(1) << "," << t_wc(2) << "," << q_wc.x() << "," << q_wc.y() << "," << q_wc.z() << "," << q_wc.w()
             << ",camera_to_world\n";

        ++image_id;
        ++exported;
    }

    fi.close();
    fcsv.close();

    {
        std::ostringstream oss;
        oss << "✅ [save_colmap] Export complete. Frames exported: " << exported << ", with last-pose fallback for " << dropped << " dropped frames";
        spdlog::debug(oss.str());
    }
    spdlog::debug("📄 [save_colmap] " + strDir + "/cameras.txt");
    spdlog::debug("📄 [save_colmap] " + strDir + "/images.txt   (COLMAP Tcw, qw qx qy qz)");
    spdlog::debug("📄 [save_colmap] " + strDir + "/points3D.txt (empty, fill with point_triangulator)");
    spdlog::debug("📄 [save_colmap] " + strDir + "/poses_debug.csv (Twc, qx qy qz qw)");
}

void system::save_point_cloud_ply(const string &filename)
{
    spdlog::debug("");
    spdlog::debug("☁️  [save_point_cloud_ply] Exporting map points to PLY");

    std::filesystem::path base_path(filename);
    std::filesystem::path out_dir = base_path.parent_path();
    std::string           stem    = base_path.stem().string();
    std::string           ext     = base_path.extension().string();

    if (stem.empty())
    {
        stem = "pointcloud";
    }
    if (ext.empty())
    {
        ext = ".ply";
    }

    if (!out_dir.empty())
    {
        std::filesystem::create_directories(out_dir);
    }

    vector<map *> vpMapsAll = _atlas->get_all_maps();
    vector<map *> vpMaps;
    vpMaps.reserve(vpMapsAll.size());
    for (map *pMap : vpMapsAll)
    {
        if (pMap && !pMap->is_bad())
        {
            vpMaps.push_back(pMap);
        }
    }

    if (vpMaps.empty())
    {
        spdlog::debug("❌ [save_point_cloud_ply] No valid maps found. Nothing to export.");
        return;
    }

    const bool multi_map = vpMaps.size() > 1;
    for (map *pMap : vpMaps)
    {
        std::filesystem::path out_file;
        if (multi_map)
        {
            std::ostringstream oss;
            oss << stem << "_map" << pMap->get_id() << ext;
            out_file = out_dir.empty() ? std::filesystem::path(oss.str()) : (out_dir / oss.str());
        }
        else
        {
            out_file = out_dir.empty() ? std::filesystem::path(stem + ext) : (out_dir / (stem + ext));
        }

        vector<map_point *> vpMPs = pMap->get_all_map_points();
        struct ColoredPoint
        {
            Eigen::Vector3f pos;
            unsigned char   r;
            unsigned char   g;
            unsigned char   b;
        };

        vector<ColoredPoint> points;
        points.reserve(vpMPs.size());

        for (map_point *pMP : vpMPs)
        {
            if (!pMP || pMP->is_bad())
            {
                continue;
            }

            Eigen::Vector3f P = pMP->get_world_position();
            if (!std::isfinite(P.x()) || !std::isfinite(P.y()) || !std::isfinite(P.z()))
            {
                continue;
            }

            const auto color = pMP->color();
            points.push_back({P, color[0], color[1], color[2]});
        }

        std::ofstream fply(out_file.string());
        if (!fply.is_open())
        {
            spdlog::debug("❌ [save_point_cloud_ply] Failed to open file: " + out_file.string());
            continue;
        }

        fply << "ply\n";
        fply << "format ascii 1.0\n";
        fply << "comment ORB_SLAM3 map point cloud\n";
        fply << "comment map_id " << pMap->get_id() << "\n";
        fply << "element vertex " << points.size() << "\n";
        fply << "property float x\n";
        fply << "property float y\n";
        fply << "property float z\n";
        fply << "property uchar red\n";
        fply << "property uchar green\n";
        fply << "property uchar blue\n";
        fply << "end_header\n";

        fply << std::fixed << std::setprecision(6);
        for (const ColoredPoint &point : points)
        {
            fply << point.pos.x() << " " << point.pos.y() << " " << point.pos.z() << " " << static_cast<int>(point.r) << " " << static_cast<int>(point.g) << " " << static_cast<int>(point.b)
                 << "\n";
        }

        fply.close();

        std::ostringstream ok;
        ok << "✅ [save_point_cloud_ply] Wrote " << points.size() << " points to " << out_file.string();
        spdlog::debug(ok.str());
    }
}

void system::save_trajectory_kitti(const string &filename)
{
    spdlog::info("Saving camera trajectory to {}", filename);
    if (_sensor == monocular)
    {
        spdlog::error("save_trajectory_kitti cannot be used for monocular");
        return;
    }

    vector<key_frame *> vpKFs = _atlas->get_all_key_frames();
    sort(vpKFs.begin(), vpKFs.end(), key_frame::compare_id);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Tow = vpKFs[0]->get_pose_inverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::key_frame *>::iterator lRit = _tracker->reference_key_frames.begin();
    list<double>::iterator                lT   = _tracker->frame_times.begin();
    for (list<Sophus::SE3f>::iterator lit = _tracker->relative_frame_poses.begin(), lend = _tracker->relative_frame_poses.end(); lit != lend; lit++, lRit++, lT++)
    {
        ORB_SLAM3::key_frame *pKF = *lRit;

        Sophus::SE3f Trw;

        if (!pKF)
        {
            continue;
        }

        while (pKF->is_bad())
        {
            Trw = Trw * pKF->pose_relative_to_parent;
            pKF = pKF->get_parent();
        }

        Trw = Trw * pKF->get_pose() * Tow;

        Sophus::SE3f    Tcw = (*lit) * Trw;
        Sophus::SE3f    Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        Eigen::Vector3f twc = Twc.translation();

        f << setprecision(9) << Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " " << Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " "
          << Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << endl;
    }
    f.close();
}

void system::save_debug_data(const int &initIdx)
{
    // 0. Save initialization trajectory
    save_trajectory_euroc("init_FrameTrajectoy_" + to_string(_local_mapper->initialization_section) + "_" + to_string(initIdx) + ".txt");

    // 1. Save scale
    ofstream f;
    f.open("init_Scale_" + to_string(_local_mapper->initialization_section) + ".txt", ios_base::app);
    f << fixed;
    f << _local_mapper->scale << endl;
    f.close();

    // 2. Save gravity direction
    f.open("init_GDir_" + to_string(_local_mapper->initialization_section) + ".txt", ios_base::app);
    f << fixed;
    f << _local_mapper->imu_world_rotation(0, 0) << "," << _local_mapper->imu_world_rotation(0, 1) << "," << _local_mapper->imu_world_rotation(0, 2) << endl;
    f << _local_mapper->imu_world_rotation(1, 0) << "," << _local_mapper->imu_world_rotation(1, 1) << "," << _local_mapper->imu_world_rotation(1, 2) << endl;
    f << _local_mapper->imu_world_rotation(2, 0) << "," << _local_mapper->imu_world_rotation(2, 1) << "," << _local_mapper->imu_world_rotation(2, 2) << endl;
    f.close();

    // 3. Save computational cost
    f.open("init_CompCost_" + to_string(_local_mapper->initialization_section) + ".txt", ios_base::app);
    f << fixed;
    f << _local_mapper->initialization_cost_time << endl;
    f.close();

    // 4. Save biases
    f.open("init_Biases_" + to_string(_local_mapper->initialization_section) + ".txt", ios_base::app);
    f << fixed;
    f << _local_mapper->gyro_bias(0) << "," << _local_mapper->gyro_bias(1) << "," << _local_mapper->gyro_bias(2) << endl;
    f << _local_mapper->accelerometer_bias(0) << "," << _local_mapper->accelerometer_bias(1) << "," << _local_mapper->accelerometer_bias(2) << endl;
    f.close();

    // 5. Save covariance matrix
    f.open("init_CovMatrix_" + to_string(_local_mapper->initialization_section) + "_" + to_string(initIdx) + ".txt", ios_base::app);
    f << fixed;
    for (int i = 0; i < _local_mapper->inertial_covariance.rows(); i++)
    {
        for (int j = 0; j < _local_mapper->inertial_covariance.cols(); j++)
        {
            if (j != 0)
            {
                f << ",";
            }
            f << setprecision(15) << _local_mapper->inertial_covariance(i, j);
        }
        f << endl;
    }
    f.close();

    // 6. Save initialization time
    f.open("init_Time_" + to_string(_local_mapper->initialization_section) + ".txt", ios_base::app);
    f << fixed;
    f << _local_mapper->initialization_time << endl;
    f.close();
}

int system::get_tracking_state()
{
    unique_lock<mutex> lock(_state_mutex);
    return _tracking_state;
}

vector<map_point *> system::get_tracked_map_points()
{
    unique_lock<mutex> lock(_state_mutex);
    return _tracked_map_points;
}

vector<keypoint> system::get_tracked_keypoints_undistorted()
{
    unique_lock<mutex> lock(_state_mutex);
    auto               keypoints = _tracked_keypoints_undistorted;
    for (auto &key : keypoints)
    {
        key.pt = key.pt_undistorted;
    }
    return keypoints;
}

double system::get_time_from_imu_init()
{
    double aux = _local_mapper->get_current_key_frame_time() - _local_mapper->first_timestamp;
    if ((aux > 0.) && _atlas->is_imu_initialized())
    {
        return _local_mapper->get_current_key_frame_time() - _local_mapper->first_timestamp;
    }
    else
    {
        return 0.f;
    }
}

bool system::is_lost()
{
    if (!_atlas->is_imu_initialized())
    {
        return false;
    }
    else
    {
        if ((_tracker->state == tracking::LOST)) //||(_tracker->state==tracking::RECENTLY_LOST))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

bool system::is_finished() { return (get_time_from_imu_init() > 0.1); }

void system::change_dataset()
{
    if (_atlas->get_current_map()->key_frame_count() < 12)
    {
        _tracker->reset_active_map();
    }
    else
    {
        _tracker->create_map_in_atlas();
    }

    _tracker->new_dataset();
}

float system::get_image_scale() { return _tracker->get_image_scale(); }

atlas *system::get_atlas() const { return _atlas.get(); }

tracking *system::get_tracker() const { return _tracker.get(); }

float system::get_camera_frame_rate() const { return _settings->fps(); }

} // namespace ORB_SLAM3
