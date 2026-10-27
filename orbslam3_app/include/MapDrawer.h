/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "ground_truth_trajectory.h"
#include "viewer_settings.h"

#include <vtkSmartPointer.h>

#include <array>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

class vtkRenderer;
class vtkActor;
class vtkPoints;
class vtkCellArray;
class vtkRenderWindow;
class vtkWindowToImageFilter;

namespace ORB_SLAM3
{
class Atlas;

class MapDrawer
{
  public:
    MapDrawer(Atlas &atlas, const viewer_settings &settings);
    ~MapDrawer();

    void         SetCurrentCameraPose(const Sophus::SE3f &camera_pose, unsigned long map_id = 0);
    void         break_camera_trajectory();
    Sophus::SE3f camera_pose();
    cv::Mat      render(int width, int height, const std::array<bool, 7> &layers);
    void         reset_view(bool top_view);
    void         follow_camera(bool top_view);
    void         navigate(float horizontal, float vertical, float zoom, bool pan);
    void         release();

  private:
    Atlas                                        &_atlas;
    viewer_settings                               _settings;
    ground_truth_trajectory                       _ground_truth;
    std::optional<Eigen::Matrix4d>                _ground_truth_alignment;
    vtkSmartPointer<vtkActor>                     _ground_truth_actor;
    vtkSmartPointer<vtkActor>                     _camera_trajectory_actor;
    vtkSmartPointer<vtkPoints>                    _camera_trajectory_points;
    vtkSmartPointer<vtkCellArray>                 _camera_trajectory_lines;
    std::vector<std::pair<Eigen::Vector3f, bool>> _pending_camera_trajectory;
    std::optional<unsigned long>                  _trajectory_map_id;
    bool                                          _connect_camera_trajectory{false};
    Sophus::SE3f                                  _camera_pose;
    Sophus::SE3f                                  _follow_pose;
    std::mutex                                    _camera_mutex;
    vtkSmartPointer<vtkRenderer>                  _renderer;
    vtkSmartPointer<vtkRenderWindow>              _window;
    vtkSmartPointer<vtkWindowToImageFilter>       _capture;
};
} // namespace ORB_SLAM3