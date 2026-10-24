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

#include "Atlas.h"
#include "viewer_settings.h"

#include <vtkSmartPointer.h>

#include <array>
#include <mutex>

#include <opencv2/core.hpp>

class vtkRenderer;
class vtkRenderWindow;
class vtkWindowToImageFilter;

namespace ORB_SLAM3
{
class MapDrawer
{
  public:
    MapDrawer(Atlas *atlas, const viewer_settings &settings);
    ~MapDrawer();
    void         SetCurrentCameraPose(const Sophus::SE3f &camera_pose);
    Sophus::SE3f camera_pose();
    cv::Mat      render(int width, int height, const std::array<bool, 5> &layers);
    void         reset_view(bool top_view);
    void         follow_camera(bool top_view);
    void         navigate(float horizontal, float vertical, float zoom, bool pan);
    void         release();

    Atlas       *mpAtlas;

  private:
    viewer_settings                         _settings;
    Sophus::SE3f                            _camera_pose;
    Sophus::SE3f                            _follow_pose;
    std::mutex                              _camera_mutex;
    vtkSmartPointer<vtkRenderer>            _renderer;
    vtkSmartPointer<vtkRenderWindow>        _window;
    vtkSmartPointer<vtkWindowToImageFilter> _capture;
};
} // namespace ORB_SLAM3