#include "viewer_settings.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <opencv2/core.hpp>

#include <yaml-cpp/yaml.h>

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

viewer_settings read_viewer_settings(const std::string &settings_file, const std::string &ground_truth_path, bool ground_truth_visible, bool correct_scale)
{
    cv::FileStorage file(settings_file, cv::FileStorage::READ);
    if (!file.isOpened())
    {
        throw std::runtime_error("Failed to open settings file for viewer: " + settings_file);
    }

    const cv::FileNode image_view_scale = file["Viewer.imageViewScale"];
    float              scale            = 1.0f;
    if (!image_view_scale.empty())
    {
        if (!image_view_scale.isReal() && !image_view_scale.isInt())
        {
            throw std::runtime_error("Invalid viewer setting: Viewer.imageViewScale");
        }
        scale = static_cast<float>(image_view_scale);
    }

    viewer_settings settings{read_required_float(file, "Viewer.KeyFrameSize"),
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
    settings.ground_truth_path          = ground_truth_path;
    settings.ground_truth_visible       = ground_truth_visible;
    settings.ground_truth_correct_scale = correct_scale;
    if (!ground_truth_path.empty())
    {
        std::ifstream stream(settings_file);
        if (stream.peek() == '%')
        {
            std::string directive;
            std::getline(stream, directive);
        }
        const auto root        = YAML::Load(stream);
        const auto calibration = root["calibration"];
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
    return settings;
}

} // namespace ORB_SLAM3
