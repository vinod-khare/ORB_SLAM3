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
        std::ofstream(path) << "%YAML:1.0\n---\n" << yaml;
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

TEST_CASE("Logging pattern supports defaults, YAML and command-line precedence", "[logging]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("logging: {}\n").logging_pattern == "%+");
    REQUIRE(settings.parse("other: value\n").logging_pattern == "%+");
    REQUIRE(settings.parse("logging:\n  pattern: '[%l] %v'\n").logging_pattern == "[%l] %v");
    REQUIRE(settings.parse("logging:\n  pattern: '[%l] %v'\n", {"--logging.pattern", "%v"}).logging_pattern == "%v");
    REQUIRE(settings.parse("logging:\n  pattern: ''\n").logging_pattern.empty());
    REQUIRE_THROWS_WITH(settings.parse("logging: invalid\n"), "Settings entry logging must be a map");
    REQUIRE_THROWS_WITH(settings.parse("logging:\n  pattern: [invalid]\n"), "Settings entry logging.pattern must be a string");
    REQUIRE_THROWS_WITH(settings.parse("logging:\n  pattern: null\n"), "Settings entry logging.pattern must be a string");
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
    REQUIRE(settings.parse("other: value\n").dbow_version == 2);
    REQUIRE(settings.parse("dbow_version: 2\n").dbow_version == 2);
    REQUIRE(settings.parse("dbow_version: 3\n").dbow_version == 3);
    const auto cli = settings.parse("dbow_version: 3\n", {"--dbow_version", "2"});
    REQUIRE(cli.dbow_version == 2);
    REQUIRE(cli.overrides.dbow_version == 2);
    REQUIRE(settings.parse("dbow_version: 2\n", {"--dbow_version=3"}).overrides.dbow_version == 3);
    REQUIRE_THROWS_WITH(settings.parse("dbow_version: 4\n"), "dbow_version must be 2 or 3");
    REQUIRE_THROWS_WITH(settings.parse("dbow_version: 2\n", {"--dbow_version", "0"}), "dbow_version must be 2 or 3");
    REQUIRE_THROWS(settings.parse("dbow_version: [3]\n"));
    REQUIRE_THROWS(settings.parse("dbow_version: null\n"));
    REQUIRE_THROWS(settings.parse("dbow_version: 2.5\n"));
    REQUIRE(settings.parse("other: value\n", {"--dbow_version", "invalid"}).exit_code == 1);
}

TEST_CASE("Ground truth options support YAML, CLI overrides and relative paths", "[ground_truth_options]")
{
    const logging_settings settings;
    const auto             defaults = settings.parse("other: value\n");
    REQUIRE(defaults.ground_truth_path.empty());
    REQUIRE_FALSE(defaults.ground_truth_visible);
    const auto yaml = settings.parse("ground_truth:\n  path: mocap/data.csv\n  visible: true\n");
    REQUIRE(yaml.ground_truth_path == (settings.directory / "mocap/data.csv").string());
    REQUIRE(yaml.ground_truth_visible);
    const auto cli = settings.parse("ground_truth:\n  path: mocap/data.csv\n  visible: true\n", {"--ground_truth.path", "trajectory.txt", "--ground_truth.visible=false"});
    REQUIRE(cli.ground_truth_path == "trajectory.txt");
    REQUIRE_FALSE(cli.ground_truth_visible);
    REQUIRE(settings.parse("ground_truth: {}\n", {"--ground_truth.path", "poses.txt", "--ground_truth.visible"}).ground_truth_visible);
    REQUIRE_THROWS_WITH(settings.parse("ground_truth: invalid\n"), "Settings entry ground_truth must be a map");
    REQUIRE_THROWS(settings.parse("ground_truth: {path: [], visible: true}\n"));
    REQUIRE_THROWS(settings.parse("ground_truth: {visible: invalid}\n"));
    REQUIRE_THROWS_WITH(settings.parse("ground_truth: {visible: true}\n"), "ground_truth.visible requires ground_truth.path");
}

TEST_CASE("Camera trajectory visibility supports defaults, YAML and CLI", "[camera_trajectory_options]")
{
    const logging_settings settings;
    REQUIRE(settings.parse("other: value\n").camera_trajectory_visible);
    REQUIRE_FALSE(settings.parse("camera_trajectory: {visible: false}\n").camera_trajectory_visible);
    REQUIRE(settings.parse("camera_trajectory: {visible: false}\n", {"--camera_trajectory.visible"}).camera_trajectory_visible);
    REQUIRE_FALSE(settings.parse("camera_trajectory: {visible: true}\n", {"--camera_trajectory.visible=false"}).camera_trajectory_visible);
    REQUIRE_THROWS_WITH(settings.parse("camera_trajectory: invalid\n"), "Settings entry camera_trajectory must be a map");
    REQUIRE_THROWS(settings.parse("camera_trajectory: {visible: invalid}\n"));
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
