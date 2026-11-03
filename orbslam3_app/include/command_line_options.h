#pragma once

#include "feature_overrides.h"
#include "folder_reader.h"
#include "viewer_settings.h"

#include <optional>
#include <string>

#include <spdlog/common.h>

class command_line_options
{
  public:
    command_line_options(int argc, char **argv);

    std::optional<int> exit_code;

    std::string                  vocab_path;
    std::string                  settings_path;
    spdlog::level::level_enum    logging_level = spdlog::level::info;
    ORB_SLAM3::viewer_settings   viewer;
    folder_reader::settings      input;
    std::optional<std::string>   output_filename;
    std::optional<std::string>   output_folder;
    bool                         save_colmap = false;
    bool                         save_ply    = false;
    bool                         use_viewer  = true;
    std::string                  slam_type   = "mono";
    ORB_SLAM3::feature_overrides overrides;
};