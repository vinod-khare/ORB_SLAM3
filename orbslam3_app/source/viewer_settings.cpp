#include "viewer_settings.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace ORB_SLAM3
{
namespace
{

float read_required_float(const YAML::Node &root, const std::string &key)
{
    const auto node = root[key];
    if (!node || !node.IsScalar())
    {
        throw std::runtime_error("Missing or invalid viewer setting: " + key);
    }
    const auto value = node.as<float>();
    if (!std::isfinite(value))
    {
        throw std::runtime_error("Viewer setting must be finite: " + key);
    }
    return value;
}

} // namespace

viewer_settings read_viewer_settings(const std::string &settings_file, const std::string &ground_truth_path, bool ground_truth_visible, bool correct_scale)
{
    std::ifstream stream(settings_file);
    if (!stream)
    {
        throw std::runtime_error("Failed to open settings file for viewer: " + settings_file);
    }
    if (stream.peek() == '%')
    {
        std::string directive;
        std::getline(stream, directive);
    }
    viewer_settings settings;
    settings.ground_truth_path          = ground_truth_path;
    settings.ground_truth_visible       = ground_truth_visible;
    settings.ground_truth_correct_scale = correct_scale;
    read_viewer_settings(settings, YAML::Load(stream), settings_file);
    return settings;
}

void read_viewer_settings(viewer_settings &settings, const YAML::Node &root, const std::string &settings_file)
{
    const auto image_view_scale = root["Viewer.imageViewScale"];
    float      scale            = 1.0f;
    if (image_view_scale)
    {
        if (!image_view_scale.IsScalar())
        {
            throw std::runtime_error("Invalid viewer setting: Viewer.imageViewScale");
        }
        scale = image_view_scale.as<float>();
        if (!std::isfinite(scale) || scale <= 0.0f)
        {
            throw std::runtime_error("Viewer.imageViewScale must be a positive finite number");
        }
    }

    settings.key_frame_size       = read_required_float(root, "Viewer.KeyFrameSize");
    settings.key_frame_line_width = read_required_float(root, "Viewer.KeyFrameLineWidth");
    settings.graph_line_width     = read_required_float(root, "Viewer.GraphLineWidth");
    settings.point_size           = read_required_float(root, "Viewer.PointSize");
    settings.camera_size          = read_required_float(root, "Viewer.CameraSize");
    settings.camera_line_width    = read_required_float(root, "Viewer.CameraLineWidth");
    settings.viewpoint_x          = read_required_float(root, "Viewer.ViewpointX");
    settings.viewpoint_y          = read_required_float(root, "Viewer.ViewpointY");
    settings.viewpoint_z          = read_required_float(root, "Viewer.ViewpointZ");
    settings.viewpoint_f          = read_required_float(root, "Viewer.ViewpointF");
    settings.image_view_scale     = scale;
    if (!settings.ground_truth_path.empty())
    {
        const auto calibration = root["data"] ? root["data"]["calibration"] : YAML::Node{};
        if (calibration && calibration["file"])
        {
            const auto calibration_path = std::filesystem::path(settings_file).parent_path() / calibration["file"].as<std::string>();
            const auto matrix           = YAML::LoadFile(calibration_path.string())["cam0"]["T_cam_imu"];
            if (matrix)
            {
                Eigen::Matrix4d transform;
                if (!matrix.IsSequence() || matrix.size() != 4)
                {
                    throw std::runtime_error("Ground truth requires a valid calibration cam0.T_cam_imu matrix");
                }
                for (std::size_t row = 0; row < 4; ++row)
                {
                    if (!matrix[row].IsSequence() || matrix[row].size() != 4)
                    {
                        throw std::runtime_error("Ground truth requires a valid calibration cam0.T_cam_imu matrix");
                    }
                    for (std::size_t column = 0; column < 4; ++column)
                    {
                        transform(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) = matrix[row][column].as<double>();
                    }
                }
                const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
                if (!transform.allFinite() || !transform.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-6) ||
                    !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-4) || std::abs(rotation.determinant() - 1.0) > 1e-4)
                {
                    throw std::runtime_error("Ground truth requires a rigid calibration cam0.T_cam_imu transform");
                }
                settings.ground_truth_body_to_camera = Sophus::SE3d(Eigen::Quaterniond(rotation).normalized(), transform.topRightCorner<3, 1>()).inverse();
            }
        }
    }
}

} // namespace ORB_SLAM3
