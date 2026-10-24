#pragma once

#include "feature_overrides.h"

#include <optional>
#include <string>

class command_line_options
{
  public:
    command_line_options(int argc, char **argv);

    std::optional<int>           exit_code;

    std::string                  vocab_path;
    std::string                  settings_path;
    std::string                  logging_pattern = "%+";
    std::string                  ground_truth_path;
    bool                         ground_truth_visible      = false;
    bool                         camera_trajectory_visible = true;
    std::optional<std::string>   data_root;
    std::optional<std::string>   data_left;
    std::optional<std::string>   data_right;
    std::optional<std::string>   data_imu;
    std::optional<std::string>   output_filename;
    std::optional<std::string>   output_folder;
    int                          frames_skip   = 0;
    int                          frames_stride = 1;
    int                          frames_take   = 0;
    bool                         save_colmap   = false;
    bool                         save_ply      = false;
    bool                         use_viewer    = true;
    std::string                  slam_type     = "mono";
    ORB_SLAM3::feature_overrides overrides;
};