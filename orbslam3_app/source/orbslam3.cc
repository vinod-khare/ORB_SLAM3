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
#include "folder_reader.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>

using namespace std;
namespace po = boost::program_options;

namespace
{
struct data_paths
{
    std::string left;
    std::string right;
    std::string imu;
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

data_paths read_data_paths(const std::string &settings_path, const po::variables_map &options, bool stereo, bool inertial)
{
    cv::FileStorage settings(settings_path, cv::FileStorage::READ);
    if (!settings.isOpened())
    {
        throw std::runtime_error("Failed to open settings file: " + settings_path);
    }

    const std::filesystem::path settings_directory = std::filesystem::absolute(settings_path).parent_path();
    const bool                  root_from_cli      = options.count("data.root") != 0;
    const std::string           root_value         = root_from_cli ? options["data.root"].as<std::string>() : read_data_setting(settings, "root");
    const std::filesystem::path root               = root_value.empty() ? settings_directory : resolve_path(root_from_cli ? std::filesystem::current_path() : settings_directory, root_value);

    const auto                  read_path          = [&](const std::string &name)
    {
        const std::string option_name = "data." + name;

        if (options.count(option_name))
        {
            return options[option_name].as<std::string>();
        }

        return read_data_setting(settings, name);
    };

    const std::string left_value  = read_path("left");
    const std::string right_value = read_path("right");
    const std::string imu_value   = read_path("imu");

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
        // TODO: refactor options parsing into a separate class
        po::options_description desc("ORB-SLAM3 TUM-VI Example - Monocular / Stereo Modes\n\nUsage options");
        desc.add_options()("help,h", "Show this help message")("vocab,v", po::value<string>()->required(), "Path to ORB vocabulary file")("settings,s", po::value<string>()->required(),
                                                                                                                                          "Path to settings YAML file")(
            "data.root", po::value<string>(), "Dataset root folder")("data.left", po::value<string>(), "Left image folder, absolute or relative to data.root")(
            "data.right", po::value<string>(), "Right image folder, absolute or relative to data.root")("data.imu", po::value<string>(), "IMU CSV file, absolute or relative to data.root")(
            "times-file,t", po::value<string>(), "Optional timestamps file. If omitted, filename stems are used as timestamps")("timestamps-type", po::value<string>()->default_value("auto"),
                                                                                                                                "Timestamps file format: auto, filename_ns, timestamp_ns, utc")(
            "output,o", po::value<string>(), "Output filename for trajectory (default: CameraTrajectory.txt)")(
            "output-folder", po::value<string>(), "Folder to write trajectory files into (created if needed)")("save-colmap", po::bool_switch()->default_value(false),
                                                                                                               "Save COLMAP-compatible sparse model output")(
            "save-ply", po::bool_switch()->default_value(false), "Save map points as PLY point cloud")("slam-type", po::value<string>()->default_value("mono"),
                                                                                                       "SLAM sensor type: mono, mono-inertial, stereo or stereo-inertial")(
            "frames-skip", po::value<int>()->default_value(0), "Number of frames to skip at the beginning")("frames-stride", po::value<int>()->default_value(1),
                                                                                                            "Take every Nth frame (stride)")("frames-take", po::value<int>()->default_value(0),
                                                                                                                                             "Maximum number of frames to process (0 = all)");

        po::positional_options_description p;
        p.add("vocab", 1);
        p.add("settings", 1);
        p.add("times-file", 1);

        po::variables_map vm;

        try
        {
            po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

            if (vm.count("help"))
            {
                cout << desc << "\nExample:\n"
                     << "  " << argv[0] << " vocab.txt settings.yaml\n"
                     << "  " << argv[0] << " --vocab vocab.txt --settings settings.yaml --data.root dataset --data.left cam0/data\n"
                     << endl;
                return 0;
            }

            po::notify(vm);
        }
        catch (po::error &e)
        {
            cerr << "ERROR: " << e.what() << endl << endl;
            cerr << "Usage: " << argv[0] << " VOCABULARY_FILE SETTINGS_FILE [TIMES_FILE] [OPTIONS]\n\n";
            cerr << "If TIMES_FILE is omitted, image filename stems are interpreted "
                    "as timestamps.\n\n";
            cerr << desc << endl;
            return 1;
        }

        string vocab_path    = vm["vocab"].as<string>();
        string settings_path = vm["settings"].as<string>();
        string times_file;
        if (vm.count("times-file"))
        {
            times_file = vm["times-file"].as<string>();
        }

        string output_folder;
        if (vm.count("output-folder"))
        {
            output_folder = vm["output-folder"].as<string>();
            if (!std::filesystem::exists(output_folder))
            {
                std::filesystem::create_directories(output_folder);
                cout << "Created output folder: " << output_folder << endl;
            }
            output_folder += "/";
        }

        string output_filename;
        bool   bFileName = false;
        if (vm.count("output"))
        {
            output_filename = vm["output"].as<string>();
            bFileName       = true;
            cout << "Output filename: " << output_filename << endl;
        }

        int                            frames_skip     = vm["frames-skip"].as<int>();
        int                            frames_stride   = vm["frames-stride"].as<int>();
        int                            frames_take     = vm["frames-take"].as<int>();
        folder_reader::timestamps_type timestamps_type = folder_reader::parse_timestamps_type(vm["timestamps-type"].as<string>());
        bool                           save_colmap     = vm["save-colmap"].as<bool>();
        bool                           save_ply        = vm["save-ply"].as<bool>();
        const string                   slam_type       = vm["slam-type"].as<string>();

        if (slam_type != "mono" && slam_type != "mono-inertial" && slam_type != "stereo" && slam_type != "stereo-inertial")
        {
            cerr << "ERROR: Invalid slam type: " << slam_type << ". Expected mono, mono-inertial, stereo or stereo-inertial" << endl;
            return 1;
        }
        const bool       stereo          = slam_type == "stereo" || slam_type == "stereo-inertial";
        const bool       inertial        = slam_type == "mono-inertial" || slam_type == "stereo-inertial";
        const data_paths paths           = read_data_paths(settings_path, vm, stereo, inertial);
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

        if (!times_file.empty())
        {
            cout << "Loading sequence: " << left_image_dir << " with times from " << times_file << "...";
        }
        else
        {
            cout << "Loading sequence: " << left_image_dir << " using filename nanoseconds as timestamps...";
        }

        const folder_reader::settings reader_settings{left_image_dir, right_image_dir, times_file, frames_skip, frames_stride, frames_take, timestamps_type, imu_csv_path};
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
        ORB_SLAM3::System  SLAM(vocab_path, settings_path, sensor, true, 0, output_filename);
        float              imageScale    = SLAM.GetImageScale();

        int                proccIm       = 0;
        // Main loop
        // cv::Mat im{};
        cv::Ptr<cv::CLAHE> clahe         = cv::createCLAHE(3.0, cv::Size(8, 8));
        const auto         to_gray_clahe = [&clahe](cv::Mat &img)
        {
            if (img.channels() == 3)
            {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }
            clahe->apply(img, img);
        };
        for (int ni = 0; ni < nImages; ni++, proccIm++)
        {
            std::cout << "Image: " << ni << "/" << nImages << "\r";
            std::cout.flush();

            // Read image/timestamp pair from sequence cursor
            cv::Mat                       image;
            cv::Mat                       image_right;
            double                        timestamp;
            vector<ORB_SLAM3::IMU::Point> imu_measurements;
            if (stereo && inertial)
            {
                auto frame       = reader.read<orbslam3::frame_stereo_inertial>();
                image            = frame.image_left;
                image_right      = frame.image_right;
                timestamp        = frame.timestamp;
                imu_measurements = std::move(frame.imu);
            }
            else if (stereo)
            {
                auto frame  = reader.read<orbslam3::frame_stereo>();
                image       = frame.image_left;
                image_right = frame.image_right;
                timestamp   = frame.timestamp;
            }
            else if (inertial)
            {
                auto frame       = reader.read<orbslam3::frame_mono_inertial>();
                image            = frame.image;
                timestamp        = frame.timestamp;
                imu_measurements = std::move(frame.imu);
            }
            else
            {
                auto frame = reader.read<orbslam3::frame_mono>();
                image      = frame.image;
                timestamp  = frame.timestamp;
            }

            auto  &im     = image;
            double tframe = timestamp;

            if (im.empty() || (stereo && image_right.empty()))
            {
                cerr << endl << "Failed to load image at: " << reader.image_path(ni) << (stereo ? " (or its right pair)" : "") << endl;
                return 1;
            }

            if (imageScale != 1.f)
            {
                int width  = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
                if (stereo)
                {
                    cv::resize(image_right, image_right, cv::Size(width, height));
                }
            }

            // Stereo tracking does not consume the color override, so only mono sets it.
            if (stereo)
            {
                to_gray_clahe(im);
                to_gray_clahe(image_right);
            }
            else
            {
                cv::Mat im_color = im.clone();
                to_gray_clahe(im);
                SLAM.SetNextFrameColor(im_color);
            }

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            // Pass the image to the SLAM system
            if (stereo)
            {
                SLAM.TrackStereo(im, image_right, tframe, imu_measurements, reader.image_path(ni));
            }
            else
            {
                SLAM.TrackMonocular(im, tframe, imu_measurements, reader.image_path(ni));
            }

            std::chrono::steady_clock::time_point t2      = std::chrono::steady_clock::now();

            double                                ttrack  = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
            ttrack_tot                                   += ttrack;

            vTimesTrack[ni]                               = ttrack;

            // Wait to load the next frame
            double T                                      = 0;
            if (ni < nImages - 1)
            {
                T = reader.timestamp(ni + 1) - tframe;
            }
            else if (ni > 0)
            {
                T = tframe - reader.timestamp(ni - 1);
            }

            if (ttrack < T)
            {
                usleep((T - ttrack) * 1e6);
            }
        }

        // Stop all threads
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
        sort(vTimesTrack.begin(), vTimesTrack.end());
        float totaltime = 0;
        for (int ni = 0; ni < nImages; ni++)
        {
            totaltime += vTimesTrack[ni];
        }
        cout << "-------" << endl << endl;
        cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
        cout << "mean tracking time: " << totaltime / proccIm << endl;

        return 0;
    }
    catch (exception &e)
    {
        cerr << "FATAL ERROR: " << e.what() << endl;
        return 1;
    }
}
