#include "command_line_options.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace po = boost::program_options;

namespace
{
YAML::Node load_settings(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Failed to open settings file: " + path);
    }
    if (stream.peek() == '%')
    {
        std::string directive;
        std::getline(stream, directive);
    }
    auto root = YAML::Load(stream);
    if (!root.IsMap())
    {
        throw std::runtime_error("Settings must be a map");
    }
    return root;
}

YAML::Node find_setting(const YAML::Node &root, const std::string &key)
{
    const auto separator = key.find('.');
    if (separator == std::string::npos)
    {
        return root[key];
    }
    const auto section = key.substr(0, separator);
    const auto node    = root[section];
    if (!node)
    {
        return YAML::Node(YAML::NodeType::Undefined);
    }
    if (!node.IsMap())
    {
        throw std::runtime_error("Settings entry " + section + " must be a map");
    }
    return find_setting(node, key.substr(separator + 1));
}

template <typename VALUE_T> VALUE_T read_option(const po::variables_map &values, const YAML::Node &root, const std::string &key, VALUE_T fallback, const std::string &yaml_key = {})
{
    const auto option = values.find(key);
    if (option != values.end() && !option->second.defaulted())
    {
        return option->second.as<VALUE_T>();
    }
    const auto node = find_setting(root, yaml_key.empty() ? key : yaml_key);
    if (!node)
    {
        return fallback;
    }
    if constexpr (std::is_same_v<VALUE_T, std::string>)
    {
        if (!node.IsScalar())
        {
            throw std::runtime_error("Settings entry " + key + " must be a string");
        }
    }
    return node.as<VALUE_T>();
}
} // namespace

command_line_options::command_line_options(int argc, char **argv)
{
    po::options_description description("ORB-SLAM3 TUM-VI Example - Monocular / Stereo Modes\n\nUsage options");
    description.add_options()("help,h", "Show this help message")("vocab,v", po::value<std::string>()->required(), "Path to compressed binary ORB vocabulary (.dbow3)")(
        "settings,s", po::value<std::string>()->required(), "Path to settings YAML file")("data.root", po::value<std::string>(), "Dataset root folder")(
        "data.left", po::value<std::string>(), "Left image folder, absolute or relative to data.root")(
        "data.right", po::value<std::string>(), "Right image folder, absolute or relative to data.root")("data.imu", po::value<std::string>(),
                                                                                                         "IMU CSV file, absolute or relative to data.root")(
        "output,o", po::value<std::string>(), "Output filename for trajectory (default: CameraTrajectory.txt)")("output-folder", po::value<std::string>(),
                                                                                                                "Folder to write trajectory files into (created if needed)")(
        "save-colmap", po::value<bool>()->implicit_value(true)->default_value(save_colmap),
        "Save COLMAP-compatible sparse model output")("save-ply", po::value<bool>()->implicit_value(true)->default_value(save_ply), "Save map points as PLY point cloud")(
        "no-viewer", po::value<bool>()->implicit_value(true)->default_value(!use_viewer),
        "Run headless without the ImGui viewer")("slam-type", po::value<std::string>()->default_value(slam_type), "SLAM sensor type: mono, mono-inertial, stereo or stereo-inertial")(
        "frames-skip", po::value<int>()->default_value(input.frames_skip), "Number of frames to skip at the beginning")(
        "frames-stride", po::value<int>()->default_value(input.frames_stride), "Take every Nth frame (stride)")("frames-take", po::value<int>()->default_value(input.frames_take),
                                                                                                                "Maximum number of frames to process (0 = all)")(
        "keypoint_detector.type", po::value<std::string>(), "Keypoint detector (fast, agast, gftt, orb, brisk, sift, akaze, star, harris_laplace, msd, tbmr, surf)")(
        "keypoint_descriptor.type", po::value<std::string>(), "Keypoint descriptor (orb, sift, brisk, brief, freak, latch, beblid, teblid, daisy, vgg, boostdesc, surf)")(
        "tracking.matcher", po::value<std::string>(),
        "Descriptor matcher (brute_force, flann)")("tracking.use_optical_flow", po::value<bool>()->implicit_value(true),
                                                   "Enable or disable flow-first tracking")("dbow_version", po::value<int>(), "Place recognition backend: 2 or 3 (overrides YAML; default: 2)")(
        "logging.pattern", po::value<std::string>(), "spdlog output pattern; overrides logging.pattern in YAML (default: %+, spdlog default)")(
        "logging.level", po::value<std::string>(), "spdlog level: trace, debug, info, warning, error, critical, off (overrides YAML; default: info)")(
        "ground_truth.path", po::value<std::string>(), "Ground-truth trajectory file (TUM text or EuRoC/TUM-VI CSV)")("ground_truth.visible", po::value<bool>()->implicit_value(true),
                                                                                                                      "Initially display ground truth (true/false; default: false)")(
        "camera_trajectory.visible", po::value<bool>()->implicit_value(true), "Initially display the full camera trajectory (true/false; default: true)")(
        "autostart", po::value<bool>()->implicit_value(true), "Start SLAM immediately (true/false; overrides YAML; default: true)");

    po::positional_options_description positional;
    positional.add("vocab", 1);
    positional.add("settings", 1);

    po::variables_map values;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(description).positional(positional).run(), values);

        if (values.count("help"))
        {
            std::ostringstream help_output;
            help_output << description << "\nExample:\n"
                        << "  " << argv[0] << " vocab.txt settings.yaml\n"
                        << "  " << argv[0] << " --vocab vocab.txt --settings settings.yaml --data.root dataset --data.left cam0/data\n";
            spdlog::info("{}", help_output.str());
            exit_code = 0;
            return;
        }

        po::notify(values);
    }
    catch (const po::error &error)
    {
        std::ostringstream error_output;
        error_output << "ERROR: " << error.what() << "\n\n"
                     << "Usage: " << argv[0] << " VOCABULARY_FILE SETTINGS_FILE [OPTIONS]\n\n"
                     << description;
        spdlog::error("{}", error_output.str());
        exit_code = 1;
        return;
    }

    vocab_path    = values["vocab"].as<std::string>();
    settings_path = values["settings"].as<std::string>();

    const auto settings = load_settings(settings_path);
    for (const auto *section : {"ground_truth", "camera_trajectory", "logging"})
    {
        if (const auto node = settings[section]; node && !node.IsMap())
        {
            throw std::runtime_error(std::string("Settings entry ") + section + " must be a map");
        }
    }
    viewer.autostart = read_option(values, settings, "autostart", viewer.autostart);
    use_viewer       = !read_option(values, settings, "no-viewer", false);
    if (!viewer.autostart && !use_viewer)
    {
        throw std::runtime_error("autostart=false requires the ImGui viewer");
    }

    overrides.dbow_version = read_option(values, settings, "dbow_version", 2);

    if (*overrides.dbow_version != 2 && *overrides.dbow_version != 3)
    {
        throw std::runtime_error("dbow_version must be 2 or 3");
    }

    viewer.ground_truth_path = read_option(values, settings, "ground_truth.path", viewer.ground_truth_path);
    if (!viewer.ground_truth_path.empty() && !values.count("ground_truth.path"))
    {
        viewer.ground_truth_path = (std::filesystem::absolute(settings_path).parent_path() / viewer.ground_truth_path).lexically_normal().string();
    }
    viewer.ground_truth_visible = read_option(values, settings, "ground_truth.visible", viewer.ground_truth_visible);

    if (viewer.ground_truth_visible && viewer.ground_truth_path.empty())
    {
        throw std::runtime_error("ground_truth.visible requires ground_truth.path");
    }

    viewer.camera_trajectory_visible = read_option(values, settings, "camera_trajectory.visible", viewer.camera_trajectory_visible);
    const auto level                 = read_option(values, settings, "logging.level", std::string(spdlog::level::to_string_view(logging_level).data()));
    logging_level                    = spdlog::level::from_str(level);
    if (logging_level == spdlog::level::off && level != "off")
    {
        throw std::runtime_error("logging.level must be trace, debug, info, warning, error, critical or off");
    }
    viewer.logging_pattern = read_option(values, settings, "logging.pattern", viewer.logging_pattern);

    const auto settings_directory = std::filesystem::absolute(settings_path).parent_path();
    const auto root_value         = read_option(values, settings, "data.root", std::string{});
    const auto root      = root_value.empty() ? settings_directory : ((values.count("data.root") ? std::filesystem::current_path() : settings_directory) / root_value).lexically_normal();
    const auto read_path = [&](const std::string &key)
    {
        const auto value = read_option(values, settings, key, std::string{});
        return value.empty() ? std::string{} : (root / value).lexically_normal().string();
    };
    input.left_image_path  = read_path("data.left");
    input.right_image_path = read_path("data.right");
    input.imu_path         = read_path("data.imu");
    for (const auto *key : {"output", "output-folder"})
    {
        if (values.count(key) || find_setting(settings, key))
        {
            (std::string(key) == "output" ? output_filename : output_folder) = read_option(values, settings, key, std::string{});
        }
    }
    input.frames_skip   = read_option(values, settings, "frames-skip", input.frames_skip, "frames.skip");
    input.frames_stride = read_option(values, settings, "frames-stride", input.frames_stride, "frames.stride");
    input.frames_take   = read_option(values, settings, "frames-take", input.frames_take, "frames.take");
    if (input.frames_skip < 0 || input.frames_stride <= 0 || input.frames_take < 0)
    {
        throw std::runtime_error("Invalid frame parameters (skip and take must be >= 0, stride must be > 0)");
    }
    save_colmap = read_option(values, settings, "save-colmap", save_colmap);
    save_ply    = read_option(values, settings, "save-ply", save_ply);
    slam_type   = read_option(values, settings, "slam-type", slam_type);
    if (slam_type != "mono" && slam_type != "mono-inertial" && slam_type != "stereo" && slam_type != "stereo-inertial")
    {
        throw std::runtime_error("Invalid slam type: " + slam_type + ". Expected mono, mono-inertial, stereo or stereo-inertial");
    }
    if (slam_type == "mono" || slam_type == "mono-inertial")
    {
        input.right_image_path.clear();
    }
    if (slam_type == "mono" || slam_type == "stereo")
    {
        input.imu_path.clear();
    }
    viewer.ground_truth_correct_scale = slam_type == "mono";
    if (use_viewer)
    {
        ORB_SLAM3::read_viewer_settings(viewer, settings, settings_path);
    }
    if (values.count("keypoint_detector.type"))
    {
        overrides.detector = values["keypoint_detector.type"].as<std::string>();
    }
    if (values.count("keypoint_descriptor.type"))
    {
        overrides.descriptor = values["keypoint_descriptor.type"].as<std::string>();
    }
    if (values.count("tracking.matcher"))
    {
        overrides.matcher = values["tracking.matcher"].as<std::string>();
    }
    if (values.count("tracking.use_optical_flow"))
    {
        overrides.use_optical_flow = values["tracking.use_optical_flow"].as<bool>();
    }
}