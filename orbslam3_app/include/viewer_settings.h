#pragma once

#include <string>

#include <sophus/se3.hpp>

namespace ORB_SLAM3
{

struct viewer_settings
{
    float        key_frame_size;
    float        key_frame_line_width;
    float        graph_line_width;
    float        point_size;
    float        camera_size;
    float        camera_line_width;
    float        viewpoint_x;
    float        viewpoint_y;
    float        viewpoint_z;
    float        viewpoint_f;
    float        image_view_scale;
    std::string  ground_truth_path{};
    bool         ground_truth_visible{false};
    bool         ground_truth_correct_scale{false};
    Sophus::SE3d ground_truth_body_to_camera{};
    bool         camera_trajectory_visible{true};
};

viewer_settings read_viewer_settings(const std::string &settings_file, const std::string &ground_truth_path = {}, bool ground_truth_visible = false, bool correct_scale = false);

} // namespace ORB_SLAM3
