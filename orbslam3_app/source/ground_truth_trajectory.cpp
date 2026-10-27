#include "ground_truth_trajectory.h"

#include <rapidcsv.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace ORB_SLAM3
{
ground_truth_trajectory::ground_truth_trajectory(const std::string &path, const Sophus::SE3d &body_to_camera) : _body_to_camera(body_to_camera)
{
    if (path.empty())
    {
        return;
    }
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Cannot open ground-truth file: " + path);
    }
    std::string line;
    while (std::getline(stream, line))
    {
        const auto start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos || line[start] == '#')
        {
            continue;
        }
        const bool            csv = line.find(',') != std::string::npos;
        std::array<double, 8> values{};
        std::istringstream    row(line);
        try
        {
            if (csv)
            {
                const rapidcsv::Document document(row, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(',', true));
                for (std::size_t index = 0; index < values.size(); ++index)
                {
                    values[index] = document.GetCell<double>(index, 0);
                }
                values[0] *= 1e-9;
            }
            else
            {
                for (auto &value : values)
                {
                    if (!(row >> value))
                    {
                        throw std::runtime_error("expected timestamp, position and quaternion");
                    }
                }
                row >> std::ws;
                if (!row.eof() && row.peek() != '#')
                {
                    throw std::runtime_error("unexpected trailing fields");
                }
            }
            if (!std::ranges::all_of(values, [](double value) { return std::isfinite(value); }))
            {
                throw std::runtime_error("non-finite pose");
            }
            if (!_samples.empty() && values[0] <= _samples.back().first)
            {
                throw std::runtime_error("timestamps must be strictly increasing");
            }
            if (!_poses.empty() && csv != _body_frame)
            {
                throw std::runtime_error("mixed TUM and CSV formats");
            }
            _body_frame                    = csv;
            Eigen::Quaterniond orientation = csv ? Eigen::Quaterniond(values[4], values[5], values[6], values[7]) : Eigen::Quaterniond(values[7], values[4], values[5], values[6]);
            if (orientation.norm() < 1e-12)
            {
                throw std::runtime_error("zero quaternion");
            }
            const Sophus::SE3d pose(orientation.normalized(), Eigen::Vector3d(values[1], values[2], values[3]));
            _poses.emplace_back(values[0], pose);
            _samples.emplace_back(values[0], (csv ? pose * _body_to_camera : pose).translation());
        }
        catch (const std::exception &error)
        {
            throw std::runtime_error("Invalid ground-truth pose " + std::to_string(_samples.size() + 1) + " in " + path + ": " + error.what());
        }
    }
    if (stream.bad() || _samples.size() < 2)
    {
        throw std::runtime_error("Ground-truth file must contain at least two readable poses: " + path);
    }
}

const trajectory_samples &ground_truth_trajectory::samples() const { return _samples; }

std::optional<Sophus::SE3d> ground_truth_trajectory::pose_at(double timestamp) const
{
    if (_poses.empty() || !std::isfinite(timestamp) || timestamp < _poses.front().first || timestamp > _poses.back().first)
    {
        return std::nullopt;
    }
    const auto   upper = std::lower_bound(_poses.begin(), _poses.end(), timestamp, [](const auto &sample, double time) { return sample.first < time; });
    Sophus::SE3d pose;
    if (upper->first == timestamp)
    {
        pose = upper->second;
    }
    else
    {
        const auto      lower    = std::prev(upper);
        const auto      fraction = (timestamp - lower->first) / (upper->first - lower->first);
        Eigen::Vector3d position;
        for (Eigen::Index axis = 0; axis < position.size(); ++axis)
        {
            position[axis] = std::lerp(lower->second.translation()[axis], upper->second.translation()[axis], fraction);
        }
        pose = Sophus::SE3d(lower->second.unit_quaternion().slerp(fraction, upper->second.unit_quaternion()), position);
    }
    return _body_frame ? pose * _body_to_camera : pose;
}

std::optional<Eigen::Matrix4d> ground_truth_trajectory::alignment(const trajectory_samples &estimate, bool correct_scale) const
{
    std::vector<Eigen::Vector3d> reference;
    std::vector<Eigen::Vector3d> target;
    for (const auto &[timestamp, position] : estimate)
    {
        const auto pose = pose_at(timestamp);
        if (!pose || !position.allFinite())
        {
            continue;
        }
        reference.push_back(pose->translation());
        target.push_back(position);
    }
    if (reference.size() < 3)
    {
        return std::nullopt;
    }
    const auto       count = static_cast<Eigen::Index>(reference.size());
    Eigen::Matrix3Xd source(3, count);
    Eigen::Matrix3Xd destination(3, count);
    for (Eigen::Index index = 0; index < count; ++index)
    {
        source.col(index)      = reference[static_cast<std::size_t>(index)];
        destination.col(index) = target[static_cast<std::size_t>(index)];
    }
    const Eigen::Matrix3Xd                   centered = source.colwise() - source.rowwise().mean();
    const Eigen::JacobiSVD<Eigen::Matrix3Xd> decomposition(centered);
    const auto                               singular = decomposition.singularValues();
    if (singular[1] <= std::max(1e-9, singular[0] * 1e-6))
    {
        return std::nullopt;
    }
    const Eigen::Matrix4d transform = Eigen::umeyama(source, destination, correct_scale);
    if (!transform.allFinite() || transform.topLeftCorner<3, 3>().determinant() <= 1e-12)
    {
        return std::nullopt;
    }
    return transform;
}
} // namespace ORB_SLAM3