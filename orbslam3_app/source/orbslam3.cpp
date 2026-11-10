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

#include "Converter.h"
#include "System.h"
#include "Tracking.h"
#include "atlas.h"
#include "command_line_options.h"
#include "frame_drawer.h"
#include "frame_prefetcher.h"
#include "map_drawer.h"
#include "time_this.h"
#include "viewer.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <thread>

#include <spdlog/spdlog.h>

using namespace std;

namespace
{
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

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const command_line_options options(argc, argv);

        if (options.exit_code.has_value())
        {
            return *options.exit_code;
        }

        spdlog::set_pattern(options.viewer.logging_pattern);
        spdlog::set_level(options.logging_level);

        const string &vocab_path    = options.vocab_path;
        const string &settings_path = options.settings_path;

        string output_folder;
        if (options.output_folder.has_value())
        {
            output_folder = *options.output_folder;

            if (!std::filesystem::exists(output_folder))
            {
                std::filesystem::create_directories(output_folder);
                spdlog::info("Created output folder: {}", output_folder);
            }

            output_folder += "/";
        }

        std::ofstream statistics_csv;
        statistics_csv.exceptions(std::ios::failbit | std::ios::badbit);
        statistics_csv.open(output_folder + "statistics.csv");
        statistics_csv.imbue(std::locale::classic());
        statistics_csv << std::setprecision(std::numeric_limits<double>::max_digits10);
        statistics_csv << "frame_id,timestamp,detected_keypoints,matched_keypoints,preprocessing_ms,frontend_ms,frames,elapsed_seconds,throughput_fps,"
                          "preprocessing_total_ms,frontend_total_ms,local_mapping_jobs,local_mapping_last_ms,local_mapping_total_ms,"
                          "loop_closing_jobs,loop_closing_last_ms,loop_closing_total_ms,global_bundle_adjustment_jobs,global_bundle_adjustment_last_ms,global_bundle_adjustment_total_ms,"
                          "left_keypoints,right_keypoints,flow_candidates,flow_tracks,flow_inliers,flow_occupied_cells,flow_only,features_extracted,tracking_state,map_id,map_change_index,"
                          "extraction_ms,stereo_matching_ms,optical_flow_ms,input_wait_ms,viewer_publication_ms\n";

        const string output_filename = options.output_filename.value_or(string{});
        const bool   bFileName       = options.output_filename.has_value();

        if (bFileName)
        {
            spdlog::info("Output filename: {}", output_filename);
        }

        const bool    save_colmap = options.save_colmap;
        const bool    save_ply    = options.save_ply;
        const bool    use_viewer  = options.use_viewer;
        const string &slam_type   = options.slam_type;
        const bool    stereo      = slam_type == "stereo" || slam_type == "stereo-inertial";
        const bool    inertial    = slam_type == "mono-inertial" || slam_type == "stereo-inertial";

        if (options.input.left_image_path.empty())
        {
            throw std::runtime_error("A left image folder is required; set data.left in the settings YAML or pass --data.left");
        }

        if (stereo && options.input.right_image_path.empty())
        {
            throw std::runtime_error("Stereo modes require data.right in the settings YAML or --data.right");
        }

        if (inertial && options.input.imu_path.empty())
        {
            throw std::runtime_error("Inertial modes require data.imu in the settings YAML or --data.imu");
        }

        spdlog::info("Loading sequence: {} using filename nanoseconds as timestamps", options.input.left_image_path);

        folder_reader reader(options.input);
        spdlog::info("Loaded sequence");

        const int nImages = static_cast<int>(reader.size());
        if (nImages <= 0)
        {
            spdlog::error("Failed to load images for sequence");
            return 1;
        }

        // Create SLAM system
        const ORB_SLAM3::System::eSensor sensor =
            stereo ? (inertial ? ORB_SLAM3::System::IMU_STEREO : ORB_SLAM3::System::STEREO) : (inertial ? ORB_SLAM3::System::IMU_MONOCULAR : ORB_SLAM3::System::MONOCULAR);
        ORB_SLAM3::System SLAM(vocab_path, settings_path, sensor, 0, output_filename, options.overrides);
        const float       image_scale = SLAM.GetImageScale();

        std::unique_ptr<ORB_SLAM3::frame_drawer> frame_drawer;
        std::unique_ptr<ORB_SLAM3::map_drawer>   map_drawer;
        std::unique_ptr<ORB_SLAM3::viewer>       viewer;
        system_viewer_control                    viewer_controls(SLAM, *SLAM.GetTracker());
        std::jthread                             viewer_thread;

        if (use_viewer)
        {
            frame_drawer        = std::make_unique<ORB_SLAM3::frame_drawer>(*SLAM.GetAtlas());
            frame_drawer->_both = stereo;
            map_drawer          = std::make_unique<ORB_SLAM3::map_drawer>(*SLAM.GetAtlas(), options.viewer);
            viewer              = std::make_unique<ORB_SLAM3::viewer>(viewer_controls, *frame_drawer, *map_drawer, image_scale, options.viewer);
            viewer->set_sequence(fmt::format("Settings: {}\nMode: {}\nDataset / sequence (left images): {}\nRight images: {}\nIMU: {}\nFrame selection: skip {}, stride {}, take {} (0 = all)",
                                             settings_path,
                                             slam_type,
                                             options.input.left_image_path,
                                             stereo ? options.input.right_image_path : "not used",
                                             inertial ? options.input.imu_path : "not used",
                                             options.input.frames_skip,
                                             options.input.frames_stride,
                                             options.input.frames_take),
                                 reader.size());

            SLAM.GetTracker()->SetResetHooks(
                [&viewer, &map_drawer]
                {
                    map_drawer->break_camera_trajectory();
                    viewer->request_stop();

                    while (!viewer->is_stopped())
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(3));
                    }
                },
                [&viewer] { viewer->release(); });

            viewer_thread = std::jthread(&ORB_SLAM3::viewer::run, viewer.get());
        }

        const auto stop_viewer = [&]
        {
            if (viewer_thread.joinable())
            {
                viewer->request_finish();
                viewer_thread.join();
            }
        };

        int                                 proccIm = 0;
        std::chrono::steady_clock::duration playback_time{};

        try
        {
            const ORB_SLAM3::time_this playback_timer(playback_time);

            SLAM.statistics().start();

            frame_prefetcher prefetch(reader, stereo, inertial, image_scale, SLAM.can_precompute_optical_flow_pyramid());

            for (int ni = 0; ni < nImages; ni++, proccIm++)
            {
                if (viewer)
                {
                    while (viewer->is_paused() && !viewer->is_exit_requested())
                    {
                        viewer->process_commands();
                        std::this_thread::sleep_for(std::chrono::milliseconds(3));
                    }

                    viewer->process_commands();

                    if (viewer->is_exit_requested())
                    {
                        break;
                    }
                }

                const auto input_started = std::chrono::steady_clock::now();
                auto       frame         = prefetch.read();
                const auto input_wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - input_started).count();

                SLAM.set_next_optical_flow_pyramid(std::move(frame.optical_flow_pyramid));
                SLAM.SetNextFrameColor(frame.color, frame.color_right);

                std::chrono::steady_clock::duration frontend_time{};
                {
                    const ORB_SLAM3::time_this timer(frontend_time);

                    if (stereo)
                    {
                        SLAM.TrackStereo(frame.image, frame.image_right, frame.timestamp, frame.imu, frame.filename);
                    }
                    else
                    {
                        SLAM.TrackMonocular(frame.image, frame.timestamp, frame.imu, frame.filename);
                    }
                }

                const double frontend_ms = std::chrono::duration<double, std::milli>(frontend_time).count();
                double       viewer_publication_ms{};

                if (viewer)
                {
                    const auto publication_started = std::chrono::steady_clock::now();
                    auto      *tracker             = SLAM.GetTracker();

                    frame_drawer->update(*tracker);

                    if (tracker->mCurrentFrame.isSet() && tracker->mState == ORB_SLAM3::Tracking::OK)
                    {
                        map_drawer->set_current_camera_pose(tracker->mCurrentFrame.GetPose(), SLAM.GetAtlas()->get_current_map()->GetId());
                    }
                    else
                    {
                        map_drawer->break_camera_trajectory();
                    }

                    viewer_publication_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - publication_started).count();
                }

                SLAM.record_frame_statistics(frame.preprocessing_ms, frontend_ms);
                const auto statistics = SLAM.statistics().get_snapshot();

                if (viewer)
                {
                    const auto &current_frame = SLAM.GetTracker()->mCurrentFrame;
                    viewer->publish_statistics(
                        {.processing      = statistics,
                         .camera_to_world = current_frame.isSet() && SLAM.GetTracker()->mState == ORB_SLAM3::Tracking::OK ? std::optional{current_frame.GetPose().inverse()} : std::nullopt,
                         .sequence_frame  = static_cast<std::size_t>(ni + 1),
                         .input_wait_ms   = input_wait_ms,
                         .viewer_publication_ms = viewer_publication_ms});
                }

                statistics_csv << statistics.frame.frame_id << ',' << statistics.frame.timestamp << ',' << statistics.frame.detected_keypoints << ',' << statistics.frame.matched_keypoints
                               << ',' << statistics.frame.preprocessing_ms << ',' << statistics.frame.frontend_ms << ',' << statistics.frames << ',' << statistics.elapsed_seconds << ','
                               << statistics.throughput_fps << ',' << statistics.preprocessing_total_ms << ',' << statistics.frontend_total_ms << ',' << statistics.local_mapping.jobs << ','
                               << statistics.local_mapping.last_ms << ',' << statistics.local_mapping.total_ms << ',' << statistics.loop_closing.jobs << ',' << statistics.loop_closing.last_ms
                               << ',' << statistics.loop_closing.total_ms << ',' << statistics.global_bundle_adjustment.jobs << ',' << statistics.global_bundle_adjustment.last_ms << ','
                               << statistics.global_bundle_adjustment.total_ms << ',' << statistics.frame.left_keypoints << ',' << statistics.frame.right_keypoints << ','
                               << statistics.frame.flow_candidates << ',' << statistics.frame.flow_tracks << ',' << statistics.frame.flow_inliers << ',' << statistics.frame.flow_occupied_cells
                               << ',' << statistics.frame.flow_only << ',' << statistics.frame.features_extracted << ',' << statistics.frame.tracking_state << ',' << statistics.frame.map_id
                               << ',' << statistics.frame.map_change_index << ',' << statistics.frame.extraction_ms << ',' << statistics.frame.stereo_matching_ms << ','
                               << statistics.frame.optical_flow_ms << ',' << input_wait_ms << ',' << viewer_publication_ms << '\n';

                spdlog::info("Frame {}/{}: detected={} matched={} preprocessing={:.3f} ms frontend={:.3f} ms throughput={:.1f} FPS | backend last completed: local={:.3f} ms ({} jobs) "
                             "loop={:.3f} ms ({} jobs) GBA={:.3f} ms ({} jobs)",
                             ni + 1,
                             nImages,
                             statistics.frame.detected_keypoints,
                             statistics.frame.matched_keypoints,
                             statistics.frame.preprocessing_ms,
                             statistics.frame.frontend_ms,
                             statistics.throughput_fps,
                             statistics.local_mapping.last_ms,
                             statistics.local_mapping.jobs,
                             statistics.loop_closing.last_ms,
                             statistics.loop_closing.jobs,
                             statistics.global_bundle_adjustment.last_ms,
                             statistics.global_bundle_adjustment.jobs);
            }

            statistics_csv.close();
        }
        catch (...)
        {
            stop_viewer();
            SLAM.Shutdown();
            throw;
        }

        const double playback_seconds = std::chrono::duration<double>(playback_time).count();
        spdlog::info("Playback: {} frames in {:.3f} s ({:.1f} frames/s), including input/prefetch waits", proccIm, playback_seconds, proccIm / playback_seconds);

        // Stop all threads
        stop_viewer();
        SLAM.Shutdown();

        const auto   statistics       = SLAM.statistics().get_snapshot();
        const double processed_frames = static_cast<double>(statistics.frames);

        spdlog::info("Processing summary: {} frames, mean preprocessing={:.3f} ms, mean frontend={:.3f} ms",
                     statistics.frames,
                     processed_frames > 0.0 ? statistics.preprocessing_total_ms / processed_frames : 0.0,
                     processed_frames > 0.0 ? statistics.frontend_total_ms / processed_frames : 0.0);
        spdlog::info("Backend completed work: local={} jobs/{:.3f} ms, loop={} jobs/{:.3f} ms, GBA={} jobs/{:.3f} ms (independent workers; times overlap)",
                     statistics.local_mapping.jobs,
                     statistics.local_mapping.total_ms,
                     statistics.loop_closing.jobs,
                     statistics.loop_closing.total_ms,
                     statistics.global_bundle_adjustment.jobs,
                     statistics.global_bundle_adjustment.total_ms);

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
            spdlog::info("COLMAP export enabled; output: {}", output_folder);
            SLAM.SaveCOLMAP(output_folder);
        }

        if (save_ply)
        {
            const string ply_output = output_folder + "pointcloud.ply";
            spdlog::info("PLY export enabled; base file: {}", ply_output);
            SLAM.SavePointCloudPLY(ply_output);
        }

        return 0;
    }
    catch (exception &e)
    {
        spdlog::critical("FATAL ERROR: {}", e.what());
        return 1;
    }
}
