/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Atlas.h"
#include "Converter.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "System.h"
#include "Tracking.h"
#include "Viewer.h"
#include "command_line_options.h"
#include "frame_prefetcher.h"
#include "viewer_settings.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>

#include <opencv2/core/core.hpp>

#include <spdlog/spdlog.h>

using namespace std;

namespace
{
struct data_paths
{
    std::string left;
    std::string right;
    std::string imu;
};

class system_viewer_control final : public ORB_SLAM3::viewer_control
{
  public:
    system_viewer_control(ORB_SLAM3::System &system, ORB_SLAM3::Tracking &tracker) : _system(system), _tracker(tracker) {}

    void set_step_by_step(bool enabled) override { _tracker.SetStepByStep(enabled); }
    void process_next_frame() override { _tracker.mbStep = true; }
    void set_localization_mode(bool enabled) override
    {
        if (enabled)
        {
            _system.ActivateLocalizationMode();
        }
        else
        {
            _system.DeactivateLocalizationMode();
        }
    }
    void reset_active_map() override { _system.ResetActiveMap(); }
    int  dbow_version() const override { return _system.dbow_version(); }
    void set_dbow_version(int version) override { _system.set_dbow_version(version); }

  private:
    ORB_SLAM3::System   &_system;
    ORB_SLAM3::Tracking &_tracker;
};

std::string read_data_setting(const cv::FileStorage &settings, const std::string &name)
{
    const cv::FileNode data = settings["data"];
    if (data.empty())
    {
        return {};
    }
    if (!data.isMap())
    {
        throw std::runtime_error("Settings entry data must be a map");
    }

    const cv::FileNode node = data[name];
    if (node.empty())
    {
        return {};
    }
    if (!node.isString())
    {
        throw std::runtime_error("Settings entry data." + name + " must be a string");
    }
    return static_cast<std::string>(node);
}

std::filesystem::path resolve_path(const std::filesystem::path &root, const std::string &value)
{
    const std::filesystem::path path(value);
    return (path.is_absolute() ? path : root / path).lexically_normal();
}

data_paths read_data_paths(const std::string &settings_path, const command_line_options &options, bool stereo, bool inertial)
{
    cv::FileStorage settings(settings_path, cv::FileStorage::READ);
    if (!settings.isOpened())
    {
        throw std::runtime_error("Failed to open settings file: " + settings_path);
    }

    const std::filesystem::path settings_directory = std::filesystem::absolute(settings_path).parent_path();
    const bool                  root_from_cli      = options.data_root.has_value();
    const std::string           root_value         = root_from_cli ? *options.data_root : read_data_setting(settings, "root");
    const std::filesystem::path root               = root_value.empty() ? settings_directory : resolve_path(root_from_cli ? std::filesystem::current_path() : settings_directory, root_value);

    const auto                  read_path = [&](const std::optional<std::string> &option, const std::string &name) { return option.has_value() ? *option : read_data_setting(settings, name); };

    const std::string           left_value  = read_path(options.data_left, "left");
    const std::string           right_value = read_path(options.data_right, "right");
    const std::string           imu_value   = read_path(options.data_imu, "imu");

    if (left_value.empty())
    {
        throw std::runtime_error("A left image folder is required; set data.left in the settings YAML or pass --data.left");
    }

    if (stereo && right_value.empty())
    {
        throw std::runtime_error("Stereo modes require data.right in the settings YAML or --data.right");
    }

    if (inertial && imu_value.empty())
    {
        throw std::runtime_error("Inertial modes require data.imu in the settings YAML or --data.imu");
    }

    return {std::filesystem::canonical(resolve_path(root, left_value)).string(), stereo ? std::filesystem::canonical(resolve_path(root, right_value)).string() : std::string{},
            inertial ? resolve_path(root, imu_value).string() : std::string{}};
}
} // namespace

double ttrack_tot = 0;
int    main(int argc, char **argv)
{
    try
    {
        const command_line_options options(argc, argv);
        if (options.exit_code.has_value())
        {
            return *options.exit_code;
        }

        spdlog::set_pattern(options.logging_pattern);

        const string &vocab_path    = options.vocab_path;
        const string &settings_path = options.settings_path;

        string        output_folder;
        if (options.output_folder.has_value())
        {
            output_folder = *options.output_folder;
            if (!std::filesystem::exists(output_folder))
            {
                std::filesystem::create_directories(output_folder);
                cout << "Created output folder: " << output_folder << endl;
            }
            output_folder += "/";
        }

        const string output_filename = options.output_filename.value_or(string{});
        const bool   bFileName       = options.output_filename.has_value();
        if (bFileName)
        {
            cout << "Output filename: " << output_filename << endl;
        }

        const int     frames_skip   = options.frames_skip;
        const int     frames_stride = options.frames_stride;
        const int     frames_take   = options.frames_take;
        const bool    save_colmap   = options.save_colmap;
        const bool    save_ply      = options.save_ply;
        const bool    use_viewer    = options.use_viewer;
        const string &slam_type     = options.slam_type;

        if (slam_type != "mono" && slam_type != "mono-inertial" && slam_type != "stereo" && slam_type != "stereo-inertial")
        {
            cerr << "ERROR: Invalid slam type: " << slam_type << ". Expected mono, mono-inertial, stereo or stereo-inertial" << endl;
            return 1;
        }
        const bool       stereo          = slam_type == "stereo" || slam_type == "stereo-inertial";
        const bool       inertial        = slam_type == "mono-inertial" || slam_type == "stereo-inertial";
        const data_paths paths           = read_data_paths(settings_path, options, stereo, inertial);
        const string    &left_image_dir  = paths.left;
        const string    &right_image_dir = paths.right;
        const string    &imu_csv_path    = paths.imu;

        if (frames_skip < 0 || frames_stride <= 0 || frames_take < 0)
        {
            cerr << "ERROR: Invalid frame parameters (skip and take must be >= 0, "
                    "stride must be > 0)"
                 << endl;
            return 1;
        }

        cout << "Loading sequence: " << left_image_dir << " using filename nanoseconds as timestamps...";

        const folder_reader::settings reader_settings{left_image_dir, right_image_dir, frames_skip, frames_stride, frames_take, imu_csv_path};
        folder_reader                 reader(reader_settings);
        cout << "LOADED!" << endl;

        const int nImages = static_cast<int>(reader.size());
        if (nImages <= 0)
        {
            cerr << "ERROR: Failed to load images for sequence" << endl;
            return 1;
        }

        // Vector for tracking time statistics
        vector<float> vTimesTrack;
        vTimesTrack.resize(nImages);

        cout << endl << "-------" << endl;
        cout.precision(17);

        // Create SLAM system
        const ORB_SLAM3::System::eSensor sensor =
            stereo ? (inertial ? ORB_SLAM3::System::IMU_STEREO : ORB_SLAM3::System::STEREO) : (inertial ? ORB_SLAM3::System::IMU_MONOCULAR : ORB_SLAM3::System::MONOCULAR);
        std::unique_ptr<ORB_SLAM3::viewer_settings> ui_settings;
        if (use_viewer)
        {
            ui_settings =
                std::make_unique<ORB_SLAM3::viewer_settings>(ORB_SLAM3::read_viewer_settings(settings_path, options.ground_truth_path, options.ground_truth_visible, slam_type == "mono"));
            ui_settings->camera_trajectory_visible = options.camera_trajectory_visible;
        }
        const ORB_SLAM3::feature_overrides      overrides = options.overrides;
        ORB_SLAM3::System                       SLAM(vocab_path, settings_path, sensor, 0, output_filename, overrides);
        const float                             image_scale = SLAM.GetImageScale();

        std::unique_ptr<ORB_SLAM3::FrameDrawer> frame_drawer;
        std::unique_ptr<ORB_SLAM3::MapDrawer>   map_drawer;
        system_viewer_control                   viewer_controls(SLAM, *SLAM.GetTracker());
        std::unique_ptr<ORB_SLAM3::Viewer>      viewer;
        std::jthread                            viewer_thread;
        if (use_viewer)
        {
            frame_drawer       = std::make_unique<ORB_SLAM3::FrameDrawer>(*SLAM.GetAtlas());
            frame_drawer->both = stereo;
            map_drawer         = std::make_unique<ORB_SLAM3::MapDrawer>(*SLAM.GetAtlas(), *ui_settings);
            viewer             = std::make_unique<ORB_SLAM3::Viewer>(viewer_controls, *frame_drawer, *map_drawer, image_scale, *ui_settings);
            viewer->both       = stereo;
            SLAM.GetTracker()->SetResetHooks(
                [&viewer, &map_drawer]
                {
                    map_drawer->break_camera_trajectory();
                    viewer->RequestStop();
                    while (!viewer->isStopped())
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(3));
                    }
                },
                [&viewer] { viewer->Release(); });
            viewer_thread = std::jthread(&ORB_SLAM3::Viewer::Run, viewer.get());
        }

        const auto stop_viewer = [&]
        {
            if (viewer_thread.joinable())
            {
                viewer->RequestFinish();
                viewer_thread.join();
            }
        };

        int        proccIm        = 0;
        const auto playback_start = std::chrono::steady_clock::now();
        try
        {
            frame_prefetcher prefetch(reader, stereo, inertial, image_scale, SLAM.can_precompute_optical_flow_pyramid());
            for (int ni = 0; ni < nImages; ni++, proccIm++)
            {
                if (viewer)
                {
                    viewer->process_commands();
                    if (viewer->exit_requested())
                    {
                        break;
                    }
                }
                std::cout << "Image: " << ni << "/" << nImages << "\r";
                std::cout.flush();

                auto frame = prefetch.read();
                SLAM.set_next_optical_flow_pyramid(std::move(frame.optical_flow_pyramid));
                if (!stereo)
                {
                    SLAM.SetNextFrameColor(frame.color);
                }

                const auto tracking_start = std::chrono::steady_clock::now();
                if (stereo)
                {
                    SLAM.TrackStereo(frame.image, frame.image_right, frame.timestamp, frame.imu, frame.filename);
                }
                else
                {
                    SLAM.TrackMonocular(frame.image, frame.timestamp, frame.imu, frame.filename);
                }
                if (viewer)
                {
                    auto *tracker = SLAM.GetTracker();
                    frame_drawer->Update(tracker);
                    if (tracker->mCurrentFrame.isSet() && tracker->mState == ORB_SLAM3::Tracking::OK)
                    {
                        map_drawer->SetCurrentCameraPose(tracker->mCurrentFrame.GetPose(), SLAM.GetAtlas()->GetCurrentMap()->GetId());
                    }
                    else
                    {
                        map_drawer->break_camera_trajectory();
                    }
                }
                const auto   tracking_end  = std::chrono::steady_clock::now();

                const double ttrack        = std::chrono::duration<double>(tracking_end - tracking_start).count();
                ttrack_tot                += ttrack;
                vTimesTrack[ni]            = ttrack;
            }
        }
        catch (...)
        {
            stop_viewer();
            SLAM.Shutdown();
            throw;
        }
        const double playback_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - playback_start).count();
        spdlog::info("Playback: {} frames in {:.3f} s ({:.1f} frames/s), including input/prefetch waits", proccIm, playback_seconds, proccIm / playback_seconds);

        // Stop all threads
        stop_viewer();
        SLAM.Shutdown();

        // Save camera trajectory
        if (bFileName)
        {
            const string kf_file = output_folder + "kf_" + output_filename + ".txt";
            const string f_file  = output_folder + "f_" + output_filename + ".txt";
            SLAM.SaveTrajectoryEuRoC(f_file);
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
        }
        else
        {
            SLAM.SaveTrajectoryEuRoC(output_folder + "CameraTrajectory.txt");
            SLAM.SaveKeyFrameTrajectoryEuRoC(output_folder + "KeyFrameTrajectory.txt");
        }

        if (save_colmap)
        {
            // Save COLMAP-compatible sparse model
            cout << "\n📦 COLMAP Export\n"
                 << "   ├─ Status: enabled (--save-colmap)\n"
                 << "   └─ Output: " << output_folder << "\n"
                 << endl;
            SLAM.SaveCOLMAP(output_folder);
        }

        if (save_ply)
        {
            const string ply_output = output_folder + "pointcloud.ply";
            cout << "\n☁️  PLY Export\n"
                 << "   ├─ Status: enabled (--save-ply)\n"
                 << "   └─ Base file: " << ply_output << "\n"
                 << endl;
            SLAM.SavePointCloudPLY(ply_output);
        }

        // Tracking time statistics
        vTimesTrack.resize(proccIm);
        sort(vTimesTrack.begin(), vTimesTrack.end());
        float totaltime = 0;
        for (int ni = 0; ni < proccIm; ni++)
        {
            totaltime += vTimesTrack[ni];
        }
        cout << "-------" << endl << endl;
        if (proccIm > 0)
        {
            cout << "median tracking time: " << vTimesTrack[proccIm / 2] << endl;
            cout << "mean tracking time: " << totaltime / proccIm << endl;
        }

        return 0;
    }
    catch (exception &e)
    {
        cerr << "FATAL ERROR: " << e.what() << endl;
        return 1;
    }
}
