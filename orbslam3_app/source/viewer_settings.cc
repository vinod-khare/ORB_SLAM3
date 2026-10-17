#include "viewer_settings.h"

#include <stdexcept>

#include <opencv2/core.hpp>

namespace ORB_SLAM3
{
namespace
{

float read_required_float(const cv::FileStorage &file, const std::string &key)
{
    const cv::FileNode node = file[key];
    if (node.empty() || (!node.isReal() && !node.isInt()))
    {
        throw std::runtime_error("Missing or invalid viewer setting: " + key);
    }
    return static_cast<float>(node);
}

} // namespace

viewer_settings read_viewer_settings(const std::string &settings_file)
{
    cv::FileStorage file(settings_file, cv::FileStorage::READ);
    if (!file.isOpened())
    {
        throw std::runtime_error("Failed to open settings file for viewer: " + settings_file);
    }

    const cv::FileNode image_view_scale = file["Viewer.imageViewScale"];
    float             scale             = 1.0f;
    if (!image_view_scale.empty())
    {
        if (!image_view_scale.isReal() && !image_view_scale.isInt())
        {
            throw std::runtime_error("Invalid viewer setting: Viewer.imageViewScale");
        }
        scale = static_cast<float>(image_view_scale);
    }

    return {read_required_float(file, "Viewer.KeyFrameSize"),
            read_required_float(file, "Viewer.KeyFrameLineWidth"),
            read_required_float(file, "Viewer.GraphLineWidth"),
            read_required_float(file, "Viewer.PointSize"),
            read_required_float(file, "Viewer.CameraSize"),
            read_required_float(file, "Viewer.CameraLineWidth"),
            read_required_float(file, "Viewer.ViewpointX"),
            read_required_float(file, "Viewer.ViewpointY"),
            read_required_float(file, "Viewer.ViewpointZ"),
            read_required_float(file, "Viewer.ViewpointF"),
            scale};
}

} // namespace ORB_SLAM3
