#include "LoopClosing.h"
#include "command_line_options.h"
#include "folder_reader.h"
#include "processing_statistics.h"
#include "time_this.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

namespace
{
struct logging_settings
{
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("orbslam3_logging_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path path      = directory / "settings.yaml";

    logging_settings() { std::filesystem::create_directory(directory); }

    ~logging_settings()
    {
        std::filesystem::remove(path);
        std::filesystem::remove(directory);
    }

    command_line_options parse(const std::string &yaml, const std::vector<std::string> &extra = {}) const
    {
        std::ofstream(path) << "%YAML:1.0\n---\nViewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 1.0\nViewer.PointSize: 2.0\n"
                               "Viewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\nViewer.ViewpointX: 0.0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -1.8\nViewer.ViewpointF: 500.0\n"
                            << yaml;
        std::vector<std::string> arguments{"orbslam3_app", "vocab.txt", path.string()};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        std::vector<char *> argv;
        for (auto &argument : arguments)
        {
            argv.push_back(argument.data());
        }
        return command_line_options(static_cast<int>(argv.size()), argv.data());
    }
};
} // namespace

TEST_CASE("Application options resolve YAML settings together", "[settings]")
{
    const logging_settings file;
    const auto             settings = file.parse(
        "autostart: false\ndbow_version: 3\nground_truth:\n  path: mocap/data.csv\n  visible: true\ncamera_trajectory:\n  visible: false\nlogging:\n  level: warning\n  pattern: '[%l] %v'\n");
    REQUIRE_FALSE(settings.viewer.autostart);
    REQUIRE(settings.overrides.dbow_version == 3);
    REQUIRE(settings.viewer.ground_truth_path == (file.directory / "mocap/data.csv").string());
    REQUIRE(settings.viewer.ground_truth_visible);
    REQUIRE_FALSE(settings.viewer.camera_trajectory_visible);
    REQUIRE(settings.logging_level == spdlog::level::warn);
    REQUIRE(settings.viewer.logging_pattern == "[%l] %v");
}

TEST_CASE("Keypoint CLI overrides use the nested YAML hierarchy", "[settings][features]")
{
    const logging_settings file;
    const auto             settings = file.parse("keypoint:\n"
                                                 "  extractor:\n"
                                                 "    n_features: 100\n"
                                                 "    scale_factor: 1.2\n"
                                                 "    n_levels: 4\n"
                                                 "    initial_threshold: 20\n"
                                                 "    minimum_threshold: 7\n"
                                                 "  detector: {type: FAST}\n"
                                                 "  descriptor: {type: ORB}\n");
    REQUIRE(settings.overrides.detector.empty());
    REQUIRE(settings.overrides.descriptor.empty());
    const auto overridden = file.parse("keypoint:\n"
                                       "  extractor:\n"
                                       "    n_features: 100\n"
                                       "    scale_factor: 1.2\n"
                                       "    n_levels: 4\n"
                                       "    initial_threshold: 20\n"
                                       "    minimum_threshold: 7\n"
                                       "  detector: {type: FAST}\n"
                                       "  descriptor: {type: ORB}\n",
                                       {"--keypoint.detector.type", "SIFT", "--keypoint.descriptor.type", "BEBLID"});
    REQUIRE(overridden.overrides.detector == "SIFT");
    REQUIRE(overridden.overrides.descriptor == "BEBLID");
}

TEST_CASE("Scope timing writes elapsed duration on all exit paths", "[statistics]")
{
    using clock = std::chrono::steady_clock;
    static_assert(!std::is_copy_constructible_v<ORB_SLAM3::time_this>);
    static_assert(!std::is_move_constructible_v<ORB_SLAM3::time_this>);
    static_assert(!std::is_copy_assignable_v<ORB_SLAM3::time_this>);
    static_assert(!std::is_move_assignable_v<ORB_SLAM3::time_this>);

    clock::duration elapsed{-1};
    const auto      before = clock::now();
    {
        const ORB_SLAM3::time_this timer(elapsed);
        REQUIRE(elapsed == clock::duration{-1});
    }
    const auto after = clock::now();
    REQUIRE(elapsed >= clock::duration::zero());
    REQUIRE(elapsed <= after - before);

    elapsed                 = clock::duration{-1};
    const auto finish_early = [&elapsed]
    {
        const ORB_SLAM3::time_this timer(elapsed);
        return;
    };
    finish_early();
    REQUIRE(elapsed >= clock::duration::zero());

    elapsed         = clock::duration{-1};
    const auto fail = [&elapsed]
    {
        const ORB_SLAM3::time_this timer(elapsed);
        throw std::runtime_error("Timed operation failed");
    };
    REQUIRE_THROWS_AS(fail(), std::runtime_error);
    REQUIRE(elapsed >= clock::duration::zero());
}

TEST_CASE("Processing statistics preserve samples and measure wall-clock throughput", "[statistics]")
{
    using statistics = ORB_SLAM3::processing_statistics;
    statistics                          measurements;
    const statistics::clock::time_point started{};
    measurements.start(started);
    REQUIRE(measurements.get_snapshot().frames == 0);
    REQUIRE(measurements.get_snapshot().throughput_fps == 0.0);

    measurements.record_frame({7, 12.5, 100, 65, 2.0, 8.0}, started + std::chrono::milliseconds(50));
    const auto first = measurements.get_snapshot();
    REQUIRE(first.frame.frame_id == 7);
    REQUIRE(first.frame.timestamp == 12.5);
    REQUIRE(first.frame.detected_keypoints == 100);
    REQUIRE(first.frame.matched_keypoints == 65);
    REQUIRE(first.frame.preprocessing_ms == 2.0);
    REQUIRE(first.frame.frontend_ms == 8.0);
    REQUIRE(first.throughput_fps == 20.0);

    measurements.record_backend(statistics::backend_stage::LOCAL_MAPPING, 20.0);
    measurements.record_backend(statistics::backend_stage::LOCAL_MAPPING, 30.0);
    measurements.record_backend(statistics::backend_stage::LOOP_CLOSING, 4.0);
    measurements.record_backend(statistics::backend_stage::GLOBAL_BUNDLE_ADJUSTMENT, 60.0);
    measurements.record_frame({8, 12.55, 0, 60, 3.0, 7.0}, started + std::chrono::milliseconds(200));
    const auto second = measurements.get_snapshot();
    REQUIRE(second.frames == 2);
    REQUIRE(second.frame.detected_keypoints == 0);
    REQUIRE(second.frame.matched_keypoints == 60);
    REQUIRE(second.preprocessing_total_ms == 5.0);
    REQUIRE(second.frontend_total_ms == 15.0);
    REQUIRE(second.throughput_fps == 10.0);
    REQUIRE(second.local_mapping.jobs == 2);
    REQUIRE(second.local_mapping.last_ms == 30.0);
    REQUIRE(second.local_mapping.total_ms == 50.0);
    REQUIRE(second.loop_closing.total_ms == 4.0);
    REQUIRE(second.global_bundle_adjustment.total_ms == 60.0);
    REQUIRE(first.local_mapping.jobs == 0);
    REQUIRE(first.frames == 1);

    measurements.start(started);
    measurements.record_frame({}, started);
    REQUIRE(measurements.get_snapshot().throughput_fps == 0.0);
    REQUIRE(measurements.get_snapshot().local_mapping.jobs == 0);
}

TEST_CASE("Processing statistics safely collect concurrent backend jobs", "[statistics]")
{
    using statistics = ORB_SLAM3::processing_statistics;
    statistics        measurements;
    std::atomic<bool> consistent{true};
    {
        std::vector<std::jthread> workers;
        for (int worker = 0; worker < 4; ++worker)
        {
            workers.emplace_back(
                [&measurements, &consistent]
                {
                    for (int job = 0; job < 100; ++job)
                    {
                        measurements.record_backend(statistics::backend_stage::LOCAL_MAPPING, 2.0);
                        const auto sample = measurements.get_snapshot();
                        if (sample.local_mapping.total_ms != static_cast<double>(sample.local_mapping.jobs) * 2.0)
                        {
                            consistent = false;
                        }
                    }
                });
        }
    }
    REQUIRE(consistent.load());
    REQUIRE(measurements.get_snapshot().local_mapping.jobs == 400);
    REQUIRE(measurements.get_snapshot().local_mapping.total_ms == 800.0);
    statistics timed_measurements;
    {
        const statistics::backend_timer timer(timed_measurements, statistics::backend_stage::LOOP_CLOSING);
    }
    REQUIRE(timed_measurements.get_snapshot().loop_closing.jobs == 1);
    REQUIRE(timed_measurements.get_snapshot().loop_closing.last_ms >= 0.0);

    const auto finish_early = [&timed_measurements]
    {
        const statistics::backend_timer timer(timed_measurements, statistics::backend_stage::GLOBAL_BUNDLE_ADJUSTMENT);
        return;
    };
    finish_early();
    REQUIRE(timed_measurements.get_snapshot().global_bundle_adjustment.jobs == 1);
}

TEST_CASE("Global bundle adjustment shutdown joins its worker before returning", "[statistics]")
{
    struct loop_worker : ORB_SLAM3::LoopClosing
    {
        explicit loop_worker(ORB_SLAM3::processing_statistics &statistics) : LoopClosing(statistics, nullptr, nullptr, nullptr, true, false) {}
        void start_job()
        {
            _global_ba_thread = std::jthread(
                [this](std::stop_token stop)
                {
                    const ORB_SLAM3::processing_statistics::backend_timer timer(_statistics, ORB_SLAM3::processing_statistics::backend_stage::GLOBAL_BUNDLE_ADJUSTMENT);
                    std::mutex                                            mutex;
                    std::condition_variable_any                           ready;
                    std::unique_lock                                      lock(mutex);
                    ready.wait(lock, stop, [] { return false; });
                    const std::lock_guard completion_lock(mMutexGBA);
                });
        }
    };

    ORB_SLAM3::processing_statistics measurements;
    const bool                       explicit_stop = GENERATE(false, true);
    {
        loop_worker worker(measurements);
        worker.start_job();
        if (explicit_stop)
        {
            worker.stop_global_bundle_adjustment();
            REQUIRE(measurements.get_snapshot().global_bundle_adjustment.jobs == 1);
            REQUIRE_FALSE(worker.isRunningGBA());
            REQUIRE(worker.isFinishedGBA());
            worker.stop_global_bundle_adjustment();
        }
    }
    REQUIRE(measurements.get_snapshot().global_bundle_adjustment.jobs == 1);
}

TEST_CASE("Autostart supports defaults, YAML, command-line precedence and headless validation", "[autostart]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("other: value\n").viewer.autostart);
    REQUIRE(settings.parse("autostart: true\n").viewer.autostart);
    REQUIRE_FALSE(settings.parse("autostart: false\n").viewer.autostart);
    REQUIRE_FALSE(settings.parse("autostart: true\n", {"--autostart", "false"}).viewer.autostart);
    REQUIRE(settings.parse("autostart: false\n", {"--autostart=true"}).viewer.autostart);
    REQUIRE(settings.parse("autostart: false\n", {"--autostart"}).viewer.autostart);
    REQUIRE(settings.parse("other: value\n", {"--no-viewer"}).viewer.autostart);
    REQUIRE(settings.parse("autostart: false\n", {"--no-viewer", "--autostart=true"}).viewer.autostart);
    REQUIRE_THROWS_WITH(settings.parse("autostart: false\n", {"--no-viewer"}), "autostart=false requires the ImGui viewer");
    REQUIRE_THROWS_WITH(settings.parse("autostart: true\n", {"--no-viewer", "--autostart=false"}), "autostart=false requires the ImGui viewer");
    REQUIRE_THROWS(settings.parse("autostart: invalid\n"));
    REQUIRE_THROWS(settings.parse("autostart: []\n"));
    REQUIRE_THROWS(settings.parse("autostart: null\n"));
}

TEST_CASE("Logging pattern supports defaults, YAML and command-line precedence", "[logging]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("logging: {}\n").viewer.logging_pattern == "%+");
    REQUIRE(settings.parse("other: value\n").viewer.logging_pattern == "%+");
    REQUIRE(settings.parse("logging:\n  pattern: '[%l] %v'\n").viewer.logging_pattern == "[%l] %v");
    REQUIRE(settings.parse("logging:\n  pattern: '[%l] %v'\n", {"--logging.pattern", "%v"}).viewer.logging_pattern == "%v");
    REQUIRE(settings.parse("logging:\n  pattern: ''\n").viewer.logging_pattern.empty());
    REQUIRE_THROWS_WITH(settings.parse("logging: invalid\n"), "Settings entry logging must be a map");
    REQUIRE_THROWS_WITH(settings.parse("logging:\n  pattern: [invalid]\n"), "Settings entry logging.pattern must be a string");
    REQUIRE_THROWS_WITH(settings.parse("logging:\n  pattern: null\n"), "Settings entry logging.pattern must be a string");
}

TEST_CASE("Logging level supports defaults, YAML and independent command-line overrides", "[logging]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("other: value\n").logging_level == spdlog::level::info);
    REQUIRE(settings.parse("logging: {}\n").logging_level == spdlog::level::info);
    const auto level = GENERATE("trace", "debug", "info", "warning", "error", "critical", "off", "warn", "err");
    REQUIRE(settings.parse(std::string("logging:\n  level: '") + level + "'\n").logging_level == spdlog::level::from_str(level));
    REQUIRE(settings.parse("logging:\n  level: trace\n", {"--logging.level", level}).logging_level == spdlog::level::from_str(level));
    const auto overridden = settings.parse("logging:\n  level: error\n  pattern: '%v'\n", {"--logging.level=debug"});
    REQUIRE(overridden.logging_level == spdlog::level::debug);
    REQUIRE(overridden.viewer.logging_pattern == "%v");
    REQUIRE(settings.parse("logging:\n  level: warning\n", {"--logging.pattern", "%v"}).logging_level == spdlog::level::warn);
    REQUIRE_THROWS_WITH(settings.parse("logging:\n  level: invalid\n"), "logging.level must be trace, debug, info, warning, error, critical or off");
    REQUIRE_THROWS(settings.parse("logging: {}\n", {"--logging.level=invalid"}));
    REQUIRE_THROWS_WITH(settings.parse("logging:\n  level: [info]\n"), "Settings entry logging.level must be a string");
    REQUIRE_THROWS_WITH(settings.parse("logging:\n  level: null\n"), "Settings entry logging.level must be a string");
    REQUIRE(settings.parse("logging:\n  level: [invalid]\n", {"--logging.level=info"}).logging_level == spdlog::level::info);
}

TEST_CASE("External timestamp file options are rejected")
{
    const logging_settings settings;
    REQUIRE(settings.parse("logging: {}\n", {"--times-file", "times.txt"}).exit_code == 1);
    REQUIRE(settings.parse("logging: {}\n", {"--timestamps-type", "utc"}).exit_code == 1);
    REQUIRE(settings.parse("logging: {}\n", {"times.txt"}).exit_code == 1);
}

TEST_CASE("DBoW version supports defaults, YAML, CLI precedence and validation", "[dbow]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("other: value\n").overrides.dbow_version == 2);
    REQUIRE(settings.parse("dbow_version: 2\n").overrides.dbow_version == 2);
    REQUIRE(settings.parse("dbow_version: 3\n").overrides.dbow_version == 3);
    const auto cli = settings.parse("dbow_version: 3\n", {"--dbow_version", "2"});
    REQUIRE(cli.overrides.dbow_version == 2);
    REQUIRE(settings.parse("dbow_version: 2\n", {"--dbow_version=3"}).overrides.dbow_version == 3);
    REQUIRE_THROWS_WITH(settings.parse("dbow_version: 4\n"), "dbow_version must be 2 or 3");
    REQUIRE_THROWS_WITH(settings.parse("dbow_version: 2\n", {"--dbow_version", "0"}), "dbow_version must be 2 or 3");
    REQUIRE_THROWS(settings.parse("dbow_version: [3]\n"));
    REQUIRE_THROWS(settings.parse("dbow_version: null\n"));
    REQUIRE_THROWS(settings.parse("dbow_version: 2.5\n"));
    REQUIRE(settings.parse("other: value\n", {"--dbow_version", "invalid"}).exit_code == 1);
}

TEST_CASE("Optical-flow option supports command-line enable and disable", "[tracking]")
{
    const logging_settings settings;
    REQUIRE_FALSE(settings.parse("logging: {}\n").overrides.use_optical_flow.has_value());
    REQUIRE(settings.parse("logging: {}\n", {"--tracking.use_optical_flow"}).overrides.use_optical_flow == true);
    REQUIRE(settings.parse("logging: {}\n", {"--tracking.use_optical_flow=false"}).overrides.use_optical_flow == false);
    REQUIRE(settings.parse("logging: {}\n", {"--tracking.use_optical_flow=true"}).overrides.use_optical_flow == true);
}

TEST_CASE("Matcher option supports the tracking command-line namespace", "[tracking]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("logging: {}\n").overrides.matcher.empty());
    REQUIRE(settings.parse("logging: {}\n", {"--tracking.matcher", "flann"}).overrides.matcher == "flann");
    REQUIRE(settings.parse("logging: {}\n", {"--tracking.matcher=brute_force"}).overrides.matcher == "brute_force");
}

TEST_CASE("Ground truth options support YAML, CLI overrides and relative paths", "[ground_truth_options]")
{
    const logging_settings settings;
    const auto             defaults = settings.parse("other: value\n");
    REQUIRE(defaults.viewer.ground_truth_path.empty());
    REQUIRE_FALSE(defaults.viewer.ground_truth_visible);
    const auto yaml = settings.parse("ground_truth:\n  path: mocap/data.csv\n  visible: true\n");
    REQUIRE(yaml.viewer.ground_truth_path == (settings.directory / "mocap/data.csv").string());
    REQUIRE(yaml.viewer.ground_truth_visible);
    const auto cli = settings.parse("ground_truth:\n  path: mocap/data.csv\n  visible: true\n", {"--ground_truth.path", "trajectory.txt", "--ground_truth.visible=false"});
    REQUIRE(cli.viewer.ground_truth_path == "trajectory.txt");
    REQUIRE_FALSE(cli.viewer.ground_truth_visible);
    REQUIRE(settings.parse("ground_truth: {}\n", {"--ground_truth.path", "poses.txt", "--ground_truth.visible"}).viewer.ground_truth_visible);
    REQUIRE_THROWS_WITH(settings.parse("ground_truth: invalid\n"), "Settings entry ground_truth must be a map");
    REQUIRE_THROWS(settings.parse("ground_truth: {path: [], visible: true}\n"));
    REQUIRE_THROWS(settings.parse("ground_truth: {visible: invalid}\n"));
    REQUIRE_THROWS_WITH(settings.parse("ground_truth: {visible: true}\n"), "ground_truth.visible requires ground_truth.path");
}

TEST_CASE("Camera trajectory visibility supports defaults, YAML and CLI", "[camera_trajectory_options]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("other: value\n").viewer.camera_trajectory_visible);
    REQUIRE_FALSE(settings.parse("camera_trajectory: {visible: false}\n").viewer.camera_trajectory_visible);
    REQUIRE(settings.parse("camera_trajectory: {visible: false}\n", {"--camera_trajectory.visible"}).viewer.camera_trajectory_visible);
    REQUIRE_FALSE(settings.parse("camera_trajectory: {visible: true}\n", {"--camera_trajectory.visible=false"}).viewer.camera_trajectory_visible);
    REQUIRE_THROWS_WITH(settings.parse("camera_trajectory: invalid\n"), "Settings entry camera_trajectory must be a map");
    REQUIRE_THROWS(settings.parse("camera_trajectory: {visible: invalid}\n"));
}

TEST_CASE("Application controls support YAML and explicit command-line overrides", "[settings]")
{
    const logging_settings file;
    const std::string      yaml     = "slam-type: stereo-inertial\nno-viewer: true\nsave-colmap: true\nsave-ply: true\n"
                                      "output: trajectory.txt\noutput-folder: results\nframes: {skip: 5, stride: 3, take: 20}\n";
    const auto             settings = file.parse(yaml);
    REQUIRE(settings.slam_type == "stereo-inertial");
    REQUIRE_FALSE(settings.use_viewer);
    REQUIRE(settings.save_colmap);
    REQUIRE(settings.save_ply);
    REQUIRE(settings.output_filename == "trajectory.txt");
    REQUIRE(settings.output_folder == "results");
    REQUIRE(settings.input.frames_skip == 5);
    REQUIRE(settings.input.frames_stride == 3);
    REQUIRE(settings.input.frames_take == 20);

    const auto cli = file.parse(yaml,
                                {"--slam-type=mono",
                                 "--no-viewer=false",
                                 "--save-colmap=false",
                                 "--save-ply=false",
                                 "--output=cli.txt",
                                 "--output-folder=cli",
                                 "--frames-skip=0",
                                 "--frames-stride=1",
                                 "--frames-take=0"});
    REQUIRE(cli.slam_type == "mono");
    REQUIRE(cli.use_viewer);
    REQUIRE_FALSE(cli.save_colmap);
    REQUIRE_FALSE(cli.save_ply);
    REQUIRE(cli.output_filename == "cli.txt");
    REQUIRE(cli.output_folder == "cli");
    REQUIRE(cli.input.frames_skip == 0);
    REQUIRE(cli.input.frames_stride == 1);
    REQUIRE(cli.input.frames_take == 0);

    const auto defaults = file.parse("other: value\n");
    REQUIRE(defaults.slam_type == "mono");
    REQUIRE(defaults.use_viewer);
    REQUIRE_FALSE(defaults.save_colmap);
    REQUIRE_FALSE(defaults.save_ply);
    REQUIRE_FALSE(defaults.output_filename);
    REQUIRE_FALSE(defaults.output_folder);
    REQUIRE(defaults.input.frames_skip == 0);
    REQUIRE(defaults.input.frames_stride == 1);
    REQUIRE(defaults.input.frames_take == 0);
    REQUIRE(file.parse("other: value\n", {"--save-colmap", "--save-ply"}).save_colmap);
    REQUIRE(file.parse("other: value\n", {"--save-colmap", "--save-ply"}).save_ply);
}

TEST_CASE("Input paths resolve once with source-aware roots", "[settings]")
{
    const logging_settings file;
    const std::string      yaml     = "slam-type: stereo-inertial\ndata: {root: dataset, left: cam0/data, right: cam1/data, imu: imu0/data.csv}\n";
    const auto             settings = file.parse(yaml);
    REQUIRE(settings.input.left_image_path == (file.directory / "dataset/cam0/data").string());
    REQUIRE(settings.input.right_image_path == (file.directory / "dataset/cam1/data").string());
    REQUIRE(settings.input.imu_path == (file.directory / "dataset/imu0/data.csv").string());

    const auto root_override = file.parse(yaml, {"--data.root=cli-dataset"});
    REQUIRE(root_override.input.left_image_path == (std::filesystem::current_path() / "cli-dataset/cam0/data").string());
    REQUIRE(root_override.input.right_image_path == (std::filesystem::current_path() / "cli-dataset/cam1/data").string());
    const auto child_override = file.parse(yaml, {"--data.left=left", "--data.right=right", "--data.imu=imu.csv"});
    REQUIRE(child_override.input.left_image_path == (file.directory / "dataset/left").string());
    REQUIRE(child_override.input.right_image_path == (file.directory / "dataset/right").string());
    REQUIRE(child_override.input.imu_path == (file.directory / "dataset/imu.csv").string());

    const auto absolute = file.parse(yaml, {"--data.left=" + (file.directory / "absolute").string()});
    REQUIRE(absolute.input.left_image_path == (file.directory / "absolute").string());
    REQUIRE(file.parse("data: {left: cam0/data}\n").input.left_image_path == (file.directory / "cam0/data").string());
    REQUIRE(file.parse(yaml, {"--data.root", ""}).input.left_image_path == (file.directory / "cam0/data").string());
    REQUIRE(file.parse(yaml, {"--slam-type=mono"}).input.right_image_path.empty());
    REQUIRE(file.parse(yaml, {"--slam-type=mono"}).input.imu_path.empty());
    REQUIRE(file.parse(yaml, {"--slam-type=stereo"}).input.imu_path.empty());
    REQUIRE(file.parse(yaml, {"--slam-type=mono-inertial"}).input.right_image_path.empty());
}

TEST_CASE("Application settings validate merged values before startup", "[settings]")
{
    const logging_settings file;
    REQUIRE_THROWS(file.parse("slam-type: invalid\n"));
    REQUIRE(file.parse("slam-type: invalid\n", {"--slam-type=mono"}).slam_type == "mono");
    REQUIRE_THROWS(file.parse("frames: {skip: -1}\n"));
    REQUIRE_THROWS(file.parse("frames: {stride: 0}\n"));
    REQUIRE_THROWS(file.parse("frames: {take: -1}\n"));
    REQUIRE(file.parse("frames: {stride: invalid}\n", {"--frames-stride=2"}).input.frames_stride == 2);
    REQUIRE_THROWS(file.parse("frames: {take: null}\n"));
    REQUIRE_THROWS_WITH(file.parse("frames: []\n"), "Settings entry frames must be a map");
    REQUIRE_THROWS_WITH(file.parse("data: invalid\n"), "Settings entry data must be a map");
    REQUIRE_THROWS_WITH(file.parse("data: {left: []}\n"), "Settings entry data.left must be a string");
    REQUIRE_THROWS(file.parse("data: {root: null}\n"));
    REQUIRE_THROWS(file.parse("output: []\n"));
    REQUIRE_THROWS(file.parse("save-colmap: invalid\n"));
    REQUIRE_THROWS_WITH(file.parse("no-viewer: true\nautostart: false\n"), "autostart=false requires the ImGui viewer");
    REQUIRE(file.parse("no-viewer: true\nautostart: false\n", {"--no-viewer=false"}).use_viewer);
}

TEST_CASE("Resolved viewer settings reuse the application YAML and skip headless loading", "[settings]")
{
    const logging_settings file;
    const auto             settings = file.parse("Viewer.imageViewScale: 2.0\nlogging: {pattern: '%v'}\n");
    REQUIRE(settings.viewer.key_frame_size == 0.05f);
    REQUIRE(settings.viewer.image_view_scale == 2.0f);
    REQUIRE(settings.viewer.logging_pattern == "%v");
    REQUIRE(settings.viewer.ground_truth_correct_scale);
    REQUIRE_FALSE(file.parse("slam-type: mono-inertial\n").viewer.ground_truth_correct_scale);
    REQUIRE_FALSE(file.parse("slam-type: stereo\n").viewer.ground_truth_correct_scale);
    REQUIRE_THROWS(file.parse("Viewer.imageViewScale: 0.0\n"));
    REQUIRE_NOTHROW(file.parse("Viewer.imageViewScale: 0.0\nno-viewer: true\n"));
    REQUIRE_THROWS(file.parse("Viewer.imageViewScale: 0.0\nno-viewer: true\n", {"--no-viewer=false"}));
}

TEST_CASE("Folder readers preserve the configured logging pattern", "[logging]")
{
    const logging_settings settings;
    std::ostringstream     output;
    const auto             sink   = std::make_shared<spdlog::sinks::ostream_sink_mt>(output);
    const auto             logger = std::make_shared<spdlog::logger>("logging-test", sink);
    struct restore_logger
    {
        std::shared_ptr<spdlog::logger> original;
        ~restore_logger() { spdlog::set_default_logger(original); }
    } restore{spdlog::default_logger()};
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("CUSTOM %v");

    folder_reader::settings reader_settings;
    reader_settings.left_image_path = settings.directory.string();
    const folder_reader reader(reader_settings);
    output.str("");
    spdlog::info("message");
    REQUIRE(output.str() == "CUSTOM message\n");
}
