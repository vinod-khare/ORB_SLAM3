#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3
{
using trajectory_samples = std::vector<std::pair<double, Eigen::Vector3d>>;

class ground_truth_trajectory
{
  public:
    explicit ground_truth_trajectory(const std::string &path, const Sophus::SE3d &body_to_camera = {});
    const trajectory_samples      &samples() const;
    std::optional<Sophus::SE3d>    pose_at(double timestamp) const;
    std::optional<Eigen::Matrix4d> alignment(const trajectory_samples &estimate, bool correct_scale) const;

  private:
    trajectory_samples                           _samples;
    std::vector<std::pair<double, Sophus::SE3d>> _poses;
    Sophus::SE3d                                 _body_to_camera;
    bool                                         _body_frame{false};
};
} // namespace ORB_SLAM3