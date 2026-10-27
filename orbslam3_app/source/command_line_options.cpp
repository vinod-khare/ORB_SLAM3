#include "command_line_options.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace po = boost::program_options;

command_line_options::command_line_options(int argc, char **argv)
{
    po::options_description description("ORB-SLAM3 TUM-VI Example - Monocular / Stereo Modes\n\nUsage options");
    description.add_options()("help,h", "Show this help message")("vocab,v", po::value<std::string>()->required(),
                                                                  "Path to ORB vocabulary file")("settings,s", po::value<std::string>()->required(), "Path to settings YAML file")(
        "data.root", po::value<std::string>(), "Dataset root folder")("data.left", po::value<std::string>(), "Left image folder, absolute or relative to data.root")(
        "data.right", po::value<std::string>(), "Right image folder, absolute or relative to data.root")("data.imu", po::value<std::string>(),
                                                                                                         "IMU CSV file, absolute or relative to data.root")(
        "output,o", po::value<std::string>(), "Output filename for trajectory (default: CameraTrajectory.txt)")("output-folder", po::value<std::string>(),
                                                                                                                "Folder to write trajectory files into (created if needed)")(
        "save-colmap", po::bool_switch()->default_value(false), "Save COLMAP-compatible sparse model output")("save-ply", po::bool_switch()->default_value(false),
                                                                                                              "Save map points as PLY point cloud")(
        "no-viewer", po::bool_switch()->default_value(false), "Run headless without the ImGui viewer")("slam-type", po::value<std::string>()->default_value("mono"),
                                                                                                       "SLAM sensor type: mono, mono-inertial, stereo or stereo-inertial")(
        "frames-skip", po::value<int>()->default_value(0), "Number of frames to skip at the beginning")("frames-stride", po::value<int>()->default_value(1), "Take every Nth frame (stride)")(
        "frames-take", po::value<int>()->default_value(0), "Maximum number of frames to process (0 = all)")("keypoint_detector.type", po::value<std::string>(),
                                                                                                            "Keypoint detector (fast, agast, gftt, orb, brisk, sift, akaze)")(
        "keypoint_descriptor.type", po::value<std::string>(), "Keypoint descriptor (orb, sift, brisk)")("tracking.matcher", po::value<std::string>(),
                                                                                                        "Descriptor matcher (brute_force, flann)")(
        "tracking.use_optical_flow", po::value<bool>()->implicit_value(true), "Enable or disable flow-first tracking")("dbow_version", po::value<int>(),
                                                                                                                       "Place recognition backend: 2 or 3 (overrides YAML; default: 2)")(
        "logging.pattern", po::value<std::string>(), "spdlog output pattern; overrides logging.pattern in YAML (default: %+, spdlog default)")(
        "ground_truth.path", po::value<std::string>(), "Ground-truth trajectory file (TUM text or EuRoC/TUM-VI CSV)")("ground_truth.visible", po::value<bool>()->implicit_value(true),
                                                                                                                      "Initially display ground truth (true/false; default: false)")(
        "camera_trajectory.visible", po::value<bool>()->implicit_value(true), "Initially display the full camera trajectory (true/false; default: true)");

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

    std::ifstream stream(settings_path);
    if (!stream)
    {
        throw std::runtime_error("Failed to open settings file: " + settings_path);
    }
    // OpenCV settings use a non-standard %YAML:1.0 directive.
    if (stream.peek() == '%')
    {
        std::string directive;
        std::getline(stream, directive);
    }
    const auto root = YAML::Load(stream);
    dbow_version    = values.count("dbow_version") ? values["dbow_version"].as<int>() : root["dbow_version"] ? root["dbow_version"].as<int>() : 2;
    if (dbow_version != 2 && dbow_version != 3)
    {
        throw std::runtime_error("dbow_version must be 2 or 3");
    }
    const auto ground_truth = root["ground_truth"];
    if (ground_truth)
    {
        if (!ground_truth.IsMap())
        {
            throw std::runtime_error("Settings entry ground_truth must be a map");
        }
        if (!values.count("ground_truth.path") && ground_truth["path"])
        {
            const auto path = ground_truth["path"].as<std::string>();
            if (!path.empty())
            {
                ground_truth_path = (std::filesystem::absolute(settings_path).parent_path() / path).lexically_normal().string();
            }
        }
        if (!values.count("ground_truth.visible") && ground_truth["visible"])
        {
            ground_truth_visible = ground_truth["visible"].as<bool>();
        }
    }
    if (values.count("ground_truth.path"))
    {
        ground_truth_path = values["ground_truth.path"].as<std::string>();
    }
    if (values.count("ground_truth.visible"))
    {
        ground_truth_visible = values["ground_truth.visible"].as<bool>();
    }
    if (ground_truth_visible && ground_truth_path.empty())
    {
        throw std::runtime_error("ground_truth.visible requires ground_truth.path");
    }
    const auto camera_trajectory = root["camera_trajectory"];
    if (camera_trajectory)
    {
        if (!camera_trajectory.IsMap())
        {
            throw std::runtime_error("Settings entry camera_trajectory must be a map");
        }
        if (!values.count("camera_trajectory.visible") && camera_trajectory["visible"])
        {
            camera_trajectory_visible = camera_trajectory["visible"].as<bool>();
        }
    }
    if (values.count("camera_trajectory.visible"))
    {
        camera_trajectory_visible = values["camera_trajectory.visible"].as<bool>();
    }

    if (values.count("logging.pattern"))
    {
        logging_pattern = values["logging.pattern"].as<std::string>();
    }
    else
    {
        const auto logging = root["logging"];
        if (logging)
        {
            if (!logging.IsMap())
            {
                throw std::runtime_error("Settings entry logging must be a map");
            }
            const auto pattern = logging["pattern"];
            if (pattern)
            {
                if (!pattern.IsScalar())
                {
                    throw std::runtime_error("Settings entry logging.pattern must be a string");
                }
                logging_pattern = pattern.as<std::string>();
            }
        }
    }
    if (values.count("data.root"))
    {
        data_root = values["data.root"].as<std::string>();
    }
    if (values.count("data.left"))
    {
        data_left = values["data.left"].as<std::string>();
    }
    if (values.count("data.right"))
    {
        data_right = values["data.right"].as<std::string>();
    }
    if (values.count("data.imu"))
    {
        data_imu = values["data.imu"].as<std::string>();
    }
    if (values.count("output"))
    {
        output_filename = values["output"].as<std::string>();
    }
    if (values.count("output-folder"))
    {
        output_folder = values["output-folder"].as<std::string>();
    }

    frames_skip   = values["frames-skip"].as<int>();
    frames_stride = values["frames-stride"].as<int>();
    frames_take   = values["frames-take"].as<int>();
    save_colmap   = values["save-colmap"].as<bool>();
    save_ply      = values["save-ply"].as<bool>();
    use_viewer    = !values["no-viewer"].as<bool>();
    slam_type     = values["slam-type"].as<std::string>();
    overrides     = {values.count("keypoint_detector.type") ? values["keypoint_detector.type"].as<std::string>() : std::string{},
                 values.count("keypoint_descriptor.type") ? values["keypoint_descriptor.type"].as<std::string>() : std::string{},
                 values.count("tracking.matcher") ? values["tracking.matcher"].as<std::string>() : std::string{}, dbow_version,
                 values.count("tracking.use_optical_flow") ? std::optional<bool>(values["tracking.use_optical_flow"].as<bool>()) : std::nullopt};
}