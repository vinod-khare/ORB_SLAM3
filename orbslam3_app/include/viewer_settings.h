#ifndef ORB_SLAM3_VIEWER_SETTINGS_H
#define ORB_SLAM3_VIEWER_SETTINGS_H

#include <string>

namespace ORB_SLAM3
{

struct viewer_settings
{
    float key_frame_size;
    float key_frame_line_width;
    float graph_line_width;
    float point_size;
    float camera_size;
    float camera_line_width;
    float viewpoint_x;
    float viewpoint_y;
    float viewpoint_z;
    float viewpoint_f;
    float image_view_scale;
};

viewer_settings read_viewer_settings(const std::string &settings_file);

} // namespace ORB_SLAM3

#endif
