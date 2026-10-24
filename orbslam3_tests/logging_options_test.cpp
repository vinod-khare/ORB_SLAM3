#include "command_line_options.h"
#include "folder_reader.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>
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
