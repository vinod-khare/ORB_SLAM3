#include "command_line_options.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>

namespace po = boost::program_options;

command_line_options::command_line_options(int argc, char **argv)
{
    po::options_description description("ORB-SLAM3 TUM-VI Example - Monocular / Stereo Modes\n\nUsage options");
    description.add_options()("help,h", "Show this help message")("vocab,v", po::value<std::string>()->required(), "Path to ORB vocabulary file")(
        "settings,s", po::value<std::string>()->required(), "Path to settings YAML file")("data.root", po::value<std::string>(), "Dataset root folder")(
        "data.left", po::value<std::string>(), "Left image folder, absolute or relative to data.root")("data.right", po::value<std::string>(),
                                                                                                       "Right image folder, absolute or relative to data.root")(
        "data.imu", po::value<std::string>(), "IMU CSV file, absolute or relative to data.root")(
        "output,o", po::value<std::string>(), "Output filename for trajectory (default: CameraTrajectory.txt)")("output-folder", po::value<std::string>(),
                                                                                                                "Folder to write trajectory files into (created if needed)")(
        "save-colmap", po::bool_switch()->default_value(false), "Save COLMAP-compatible sparse model output")("save-ply", po::bool_switch()->default_value(false),
                                                                                                              "Save map points as PLY point cloud")(
        "no-viewer", po::bool_switch()->default_value(false), "Run headless without the Pangolin viewer")("slam-type", po::value<std::string>()->default_value("mono"),
                                                                                                          "SLAM sensor type: mono, mono-inertial, stereo or stereo-inertial")(
        "frames-skip", po::value<int>()->default_value(0), "Number of frames to skip at the beginning")("frames-stride", po::value<int>()->default_value(1), "Take every Nth frame (stride)")(
        "frames-take", po::value<int>()->default_value(0), "Maximum number of frames to process (0 = all)")("keypoint_detector.type", po::value<std::string>(),
                                                                                                            "Keypoint detector (fast, agast, gftt, orb, brisk, sift, akaze)")(
        "keypoint_descriptor.type", po::value<std::string>(), "Keypoint descriptor (orb, sift, brisk)")("descriptor_matcher.type", po::value<std::string>(),
                                                                                                        "Descriptor matcher (brute_force, flann)")(
        "logging.pattern", po::value<std::string>(), "spdlog output pattern; overrides logging.pattern in YAML (default: %+, spdlog default)");

    po::positional_options_description positional;
    positional.add("vocab", 1);
    positional.add("settings", 1);

    po::variables_map values;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(description).positional(positional).run(), values);

        if (values.count("help"))
        {
            std::cout << description << "\nExample:\n"
                      << "  " << argv[0] << " vocab.txt settings.yaml\n"
                      << "  " << argv[0] << " --vocab vocab.txt --settings settings.yaml --data.root dataset --data.left cam0/data\n"
                      << std::endl;
            exit_code = 0;
            return;
        }

        po::notify(values);
    }
    catch (const po::error &error)
    {
        std::cerr << "ERROR: " << error.what() << std::endl << std::endl;
        std::cerr << "Usage: " << argv[0] << " VOCABULARY_FILE SETTINGS_FILE [OPTIONS]\n\n";
        std::cerr << description << std::endl;
        exit_code = 1;
        return;
    }

    vocab_path    = values["vocab"].as<std::string>();
    settings_path = values["settings"].as<std::string>();
    
    if (values.count("logging.pattern"))
    {
        logging_pattern = values["logging.pattern"].as<std::string>();
    }
    else
    {
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
        const auto root    = YAML::Load(stream);
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

    frames_skip     = values["frames-skip"].as<int>();
    frames_stride   = values["frames-stride"].as<int>();
    frames_take     = values["frames-take"].as<int>();
    save_colmap     = values["save-colmap"].as<bool>();
    save_ply        = values["save-ply"].as<bool>();
    use_viewer      = !values["no-viewer"].as<bool>();
    slam_type       = values["slam-type"].as<std::string>();
    overrides       = {values.count("keypoint_detector.type") ? values["keypoint_detector.type"].as<std::string>() : std::string{},
                       values.count("keypoint_descriptor.type") ? values["keypoint_descriptor.type"].as<std::string>() : std::string{},
                       values.count("descriptor_matcher.type") ? values["descriptor_matcher.type"].as<std::string>() : std::string{}};
}