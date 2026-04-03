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

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include <unistd.h>
#include <filesystem>

#include<opencv2/core/core.hpp>
#include<boost/program_options.hpp>

#include"System.h"
#include "Converter.h"

using namespace std;
namespace po = boost::program_options;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps,
                int frames_skip = 0, int frames_stride = 1, int frames_take = 0);

double ttrack_tot = 0;
int main(int argc, char **argv)
{
    try {
        po::options_description desc("ORB-SLAM3 TUM-VI Example - Monocular Mode\n\nUsage options");
        desc.add_options()
            ("help,h", "Show this help message")
            ("vocab,v", po::value<string>()->required(), "Path to ORB vocabulary file")
            ("settings,s", po::value<string>()->required(), "Path to settings YAML file")
            ("sequence,d", po::value<vector<string>>()->required()->multitoken(), "Image directory and times file pairs (can be specified multiple times)")
            ("output,o", po::value<string>(), "Output filename for trajectory (default: CameraTrajectory.txt)")
            ("output-folder", po::value<string>(), "Folder to write trajectory files into (created if needed)")
            ("frames-skip", po::value<int>()->default_value(0), "Number of frames to skip at the beginning")
            ("frames-stride", po::value<int>()->default_value(1), "Take every Nth frame (stride)")
            ("frames-take", po::value<int>()->default_value(0), "Maximum number of frames to process (0 = all)")
            ;

        po::positional_options_description p;
        p.add("vocab", 1);
        p.add("settings", 1);
        p.add("sequence", -1);

        po::variables_map vm;
        
        try {
            po::store(po::command_line_parser(argc, argv)
                    .options(desc)
                    .positional(p)
                    .run(),
                vm);
            
            if (vm.count("help")) {
                cout << desc << "\nExample:\n"
                     << "  " << argv[0] << " vocab.txt settings.yaml img_dir1 times1.txt img_dir2 times2.txt --output my_traj.txt\n"
                     << "  " << argv[0] << " --vocab vocab.txt --settings settings.yaml --sequence img_dir1 times1.txt --sequence img_dir2 times2.txt\n"
                     << endl;
                return 0;
            }
            
            po::notify(vm);
        }
        catch(po::error& e) {
            cerr << "ERROR: " << e.what() << endl << endl;
            cerr << "Usage: " << argv[0] << " VOCABULARY_FILE SETTINGS_FILE IMAGE_DIR TIMES_FILE [IMAGE_DIR TIMES_FILE ...] [OPTIONS]\n\n";
            cerr << desc << endl;
            return 1;
        }

        string vocab_path = vm["vocab"].as<string>();
        string settings_path = vm["settings"].as<string>();
        vector<string> sequence_args = vm["sequence"].as<vector<string>>();
        
        // Parse sequence pairs (each pair is: image_dir times_file)
        if (sequence_args.size() % 2 != 0) {
            cerr << "ERROR: Sequence arguments must come in pairs (image_directory times_file)" << endl;
            return 1;
        }

        int num_seq = sequence_args.size() / 2;
        vector<vector<string>> vstrImageFilenames(num_seq);
        vector<vector<double>> vTimestampsCam(num_seq);
        vector<int> nImages(num_seq);

        string output_folder;
        if (vm.count("output-folder")) {
            output_folder = vm["output-folder"].as<string>();
            if (!std::filesystem::exists(output_folder)) {
                std::filesystem::create_directories(output_folder);
                cout << "Created output folder: " << output_folder << endl;
            }
            output_folder += "/";
        }

        string output_filename;
        bool bFileName = false;
        if (vm.count("output")) {
            output_filename = vm["output"].as<string>();
            bFileName = true;
            cout << "Output filename: " << output_filename << endl;
        }

        int frames_skip = vm["frames-skip"].as<int>();
        int frames_stride = vm["frames-stride"].as<int>();
        int frames_take = vm["frames-take"].as<int>();

        if (frames_skip < 0 || frames_stride <= 0 || frames_take < 0) {
            cerr << "ERROR: Invalid frame parameters (skip and take must be >= 0, stride must be > 0)" << endl;
            return 1;
        }

        int tot_images = 0;
        for (int seq = 0; seq < num_seq; seq++) {
            string image_dir = sequence_args[2 * seq];
            string times_file = sequence_args[2 * seq + 1];
            
            cout << "Loading sequence " << seq << ": " << image_dir << " with times from " << times_file << "...";
            LoadImages(image_dir, times_file, vstrImageFilenames[seq], vTimestampsCam[seq], 
                      frames_skip, frames_stride, frames_take);
            cout << "LOADED!" << endl;

            nImages[seq] = vstrImageFilenames[seq].size();
            tot_images += nImages[seq];

            if (nImages[seq] <= 0) {
                cerr << "ERROR: Failed to load images for sequence " << seq << endl;
                return 1;
            }
        }

        // Vector for tracking time statistics
        vector<float> vTimesTrack;
        vTimesTrack.resize(tot_images);

        cout << endl << "-------" << endl;
        cout.precision(17);

        // Create SLAM system
        ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true, 0, output_filename);
        float imageScale = SLAM.GetImageScale();

#ifdef REGISTER_TIMES
        double t_resize = 0.f;
        double t_track = 0.f;
#endif

        int proccIm = 0;
        for (int seq = 0; seq < num_seq; seq++) {
            // Main loop
            cv::Mat im;
            proccIm = 0;
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            for (int ni = 0; ni < nImages[seq]; ni++, proccIm++) {
                std::cout << "Sequence: " << seq << ", Image: " << ni << "/" << nImages[seq] << "\r";
                std::cout.flush();

                // Read image from file
                im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_GRAYSCALE);

                if (imageScale != 1.f) {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                    std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
                    std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #endif
#endif
                    int width = im.cols * imageScale;
                    int height = im.rows * imageScale;
                    cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                    std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
                    std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #endif
                    t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                    SLAM.InsertResizeTime(t_resize);
#endif
                }

                // clahe
                clahe->apply(im, im);

                double tframe = vTimestampsCam[seq][ni];

                if (im.empty()) {
                    cerr << endl << "Failed to load image at: " << vstrImageFilenames[seq][ni] << endl;
                    return 1;
                }

#ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

                // Pass the image to the SLAM system
                SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

#ifdef REGISTER_TIMES
                t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
                SLAM.InsertTrackTime(t_track);
#endif

                double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                ttrack_tot += ttrack;

                vTimesTrack[ni] = ttrack;

                // Wait to load the next frame
                double T = 0;
                if (ni < nImages[seq] - 1)
                    T = vTimestampsCam[seq][ni + 1] - tframe;
                else if (ni > 0)
                    T = tframe - vTimestampsCam[seq][ni - 1];

                if (ttrack < T)
                    usleep((T - ttrack) * 1e6);
            }

            if (seq < num_seq - 1) {
                cout << "Changing the dataset" << endl;
                SLAM.ChangeDataset();
            }
        }

        // Stop all threads
        SLAM.Shutdown();

        // Save camera trajectory
        if (bFileName) {
            const string kf_file = output_folder + "kf_" + output_filename + ".txt";
            const string f_file = output_folder + "f_" + output_filename + ".txt";
            SLAM.SaveTrajectoryEuRoC(f_file);
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
        } else {
            SLAM.SaveTrajectoryEuRoC(output_folder + "CameraTrajectory.txt");
            SLAM.SaveKeyFrameTrajectoryEuRoC(output_folder + "KeyFrameTrajectory.txt");
        }

        // Tracking time statistics
        sort(vTimesTrack.begin(), vTimesTrack.end());
        float totaltime = 0;
        for (int ni = 0; ni < nImages[0]; ni++) {
            totaltime += vTimesTrack[ni];
        }
        cout << "-------" << endl << endl;
        cout << "median tracking time: " << vTimesTrack[nImages[0] / 2] << endl;
        cout << "mean tracking time: " << totaltime / proccIm << endl;

        return 0;

    }
    catch(exception& e) {
        cerr << "FATAL ERROR: " << e.what() << endl;
        return 1;
    }
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps,
                int frames_skip, int frames_stride, int frames_take)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vector<string> allImages;
    vector<double> allTimestamps;
    allTimestamps.reserve(5000);
    allImages.reserve(5000);
    
    // Load all frames first
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);

        if(!s.empty())
        {
            if (s[0] == '#')
                continue;

            int pos = s.find(' ');
            string item = s.substr(0, pos);

            allImages.push_back(strImagePath + "/" + item + ".png");
            double t = stod(item);
            allTimestamps.push_back(t/1e9);
        }
    }
    
    // Apply frame filtering
    int start_idx = frames_skip;
    int count = 0;
    
    for (int i = start_idx; i < (int)allImages.size(); i += frames_stride) {
        if (frames_take > 0 && count >= frames_take) {
            break;
        }
        vstrImages.push_back(allImages[i]);
        vTimeStamps.push_back(allTimestamps[i]);
        count++;
    }
}



