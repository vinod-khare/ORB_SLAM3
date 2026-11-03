#pragma once

#include <string>

#include <sophus/se3.hpp>

namespace YAML
{
class Node;
}

namespace ORB_SLAM3
{

struct viewer_settings
{
    float        key_frame_size{};
    float        key_frame_line_width{};
    float        graph_line_width{};
    float        point_size{};
    float        camera_size{};
    float        camera_line_width{};
    float        viewpoint_x{};
    float        viewpoint_y{};
    float        viewpoint_z{};
    float        viewpoint_f{};
    float        image_view_scale{1.0f};
    std::string  ground_truth_path{};
    bool         ground_truth_visible{false};
    bool         ground_truth_correct_scale{false};
    Sophus::SE3d ground_truth_body_to_camera{};
    bool         camera_trajectory_visible{true};
    std::string  logging_pattern{"%+"};
    bool         autostart{true};
};

viewer_settings read_viewer_settings(const std::string &settings_file, const std::string &ground_truth_path = {}, bool ground_truth_visible = false, bool correct_scale = false);
void            read_viewer_settings(viewer_settings &settings, const YAML::Node &root, const std::string &settings_file);

} // namespace ORB_SLAM3
