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

#include "MapDrawer.h"

#include "KeyFrame.h"
#include "MapPoint.h"

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkWindowToImageFilter.h>

#include <algorithm>
#include <cmath>
#include <set>

namespace ORB_SLAM3
{
MapDrawer::MapDrawer(Atlas *atlas, const viewer_settings &settings) : mpAtlas(atlas), _settings(settings) {}
MapDrawer::~MapDrawer() = default;

void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &camera_pose)
{
    std::unique_lock lock(_camera_mutex, std::try_to_lock);
    if (lock.owns_lock())
    {
        _camera_pose = camera_pose.inverse();
    }
}

Sophus::SE3f MapDrawer::camera_pose()
{
    std::scoped_lock lock(_camera_mutex);
    return _camera_pose;
}

void MapDrawer::reset_view(bool top_view)
{
    if (!_renderer)
    {
        return;
    }
    auto *camera = _renderer->GetActiveCamera();
    camera->SetFocalPoint(0, 0, 0);
    if (top_view)
    {
        camera->SetPosition(0, 0.01, 50);
        camera->SetViewUp(0, 1, 0);
    }
    else
    {
        camera->SetPosition(_settings.viewpoint_x, _settings.viewpoint_y, _settings.viewpoint_z);
        camera->SetViewUp(0, -1, 0);
    }
    camera->SetViewAngle(2.0 * std::atan(768.0 / (2.0 * std::max(1.0f, _settings.viewpoint_f))) * 180.0 / std::acos(-1.0));
    _follow_pose = Sophus::SE3f{};
}

void MapDrawer::follow_camera(bool top_view)
{
    if (!_renderer)
    {
        return;
    }
    auto pose = camera_pose();
    if (top_view)
    {
        pose = Sophus::SE3f(Eigen::Quaternionf::Identity(), pose.translation());
    }
    const auto delta     = pose * _follow_pose.inverse();
    auto      *camera    = _renderer->GetActiveCamera();
    const auto transform = [&](const double *point) -> Eigen::Vector3f
    { return delta * Eigen::Vector3f(static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2])); };
    const auto            position  = transform(camera->GetPosition());
    const auto            focus     = transform(camera->GetFocalPoint());
    const auto           *up        = camera->GetViewUp();
    const Eigen::Vector3f direction = delta.rotationMatrix() * Eigen::Vector3f(static_cast<float>(up[0]), static_cast<float>(up[1]), static_cast<float>(up[2]));
    camera->SetPosition(position.x(), position.y(), position.z());
    camera->SetFocalPoint(focus.x(), focus.y(), focus.z());
    camera->SetViewUp(direction.x(), direction.y(), direction.z());
    _follow_pose = pose;
}

void MapDrawer::navigate(float horizontal, float vertical, float zoom, bool pan)
{
    if (!_renderer)
    {
        return;
    }
    auto *camera = _renderer->GetActiveCamera();
    if (pan)
    {
        const auto           *position = camera->GetPosition();
        const auto           *focus    = camera->GetFocalPoint();
        const auto           *up       = camera->GetViewUp();
        const Eigen::Vector3d direction(focus[0] - position[0], focus[1] - position[1], focus[2] - position[2]);
        const Eigen::Vector3d vertical_axis(up[0], up[1], up[2]);
        const Eigen::Vector3d offset = direction.norm() * 0.002 * (-horizontal * direction.normalized().cross(vertical_axis).normalized() + vertical * vertical_axis);
        camera->SetPosition(position[0] + offset.x(), position[1] + offset.y(), position[2] + offset.z());
        camera->SetFocalPoint(focus[0] + offset.x(), focus[1] + offset.y(), focus[2] + offset.z());
    }
    else
    {
        camera->Azimuth(-horizontal * 0.3);
        camera->Elevation(vertical * 0.3);
        camera->OrthogonalizeViewUp();
    }
    camera->Dolly(std::pow(1.15, zoom));
}

cv::Mat MapDrawer::render(int width, int height, const std::array<bool, 5> &layers)
{
    if (!_window)
    {
        _renderer = vtkSmartPointer<vtkRenderer>::New();
        _renderer->SetBackground(0.96, 0.97, 0.98);
        _window = vtkSmartPointer<vtkRenderWindow>::New();
        _window->SetOffScreenRendering(1);
        _window->SetShowWindow(false);
        _window->SetMultiSamples(0);
        _window->AddRenderer(_renderer);
        _capture = vtkSmartPointer<vtkWindowToImageFilter>::New();
        _capture->SetInput(_window);
        _capture->SetInputBufferTypeToRGB();
        _capture->ReadFrontBufferOff();
        _capture->ShouldRerenderOff();
        reset_view(false);
    }
    _window->SetSize(std::clamp(width, 1, 4096), std::clamp(height, 1, 4096));
    _renderer->RemoveAllViewProps();
    std::array<vtkSmartPointer<vtkPoints>, 14>    points;
    std::array<vtkSmartPointer<vtkCellArray>, 14> cells;
    for (std::size_t group = 0; group < points.size(); ++group)
    {
        points[group] = vtkSmartPointer<vtkPoints>::New();
        cells[group]  = vtkSmartPointer<vtkCellArray>::New();
    }
    const auto add_line = [&](std::size_t group, const Eigen::Vector3f &start, const Eigen::Vector3f &end)
    {
        const vtkIdType indices[]{points[group]->InsertNextPoint(start.x(), start.y(), start.z()), points[group]->InsertNextPoint(end.x(), end.y(), end.z())};
        cells[group]->InsertNextCell(2, indices);
    };
    const auto add_camera = [&](std::size_t group, const Sophus::SE3f &pose, float size)
    {
        const std::array<Eigen::Vector3f, 4> corners{
            {{size, size * 0.75f, size * 0.6f}, {size, -size * 0.75f, size * 0.6f}, {-size, -size * 0.75f, size * 0.6f}, {-size, size * 0.75f, size * 0.6f}}
        };
        for (std::size_t index = 0; index < corners.size(); ++index)
        {
            add_line(group, pose.translation(), pose * corners[index]);
            add_line(group, pose * corners[index], pose * corners[(index + 1) % corners.size()]);
        }
    };
    add_camera(7, camera_pose(), _settings.camera_size);
    auto *active_map = mpAtlas->GetCurrentMap();
    if (active_map)
    {
        std::set<unsigned long> optimized_keyframes;
        std::set<unsigned long> fixed_keyframes;
        if (layers[4])
        {
            std::unique_lock lock(active_map->viewer_mutex, std::try_to_lock);
            if (lock.owns_lock())
            {
                optimized_keyframes = active_map->msOptKFs;
                fixed_keyframes     = active_map->msFixedKFs;
            }
        }
        if (layers[0])
        {
            const auto                 references = active_map->GetReferenceMapPoints();
            const std::set<MapPoint *> reference_set(references.begin(), references.end());
            for (auto *point : active_map->GetAllMapPoints())
            {
                if (!point || point->isBad())
                {
                    continue;
                }
                const auto            group    = reference_set.contains(point) ? 1u : 0u;
                const Eigen::Vector3f position = point->GetWorldPos();
                const auto            index    = points[group]->InsertNextPoint(position.x(), position.y(), position.z());
                cells[group]->InsertNextCell(1, &index);
            }
        }
        for (auto *map : mpAtlas->GetAllMaps())
        {
            for (auto *keyframe : map->GetAllKeyFrames())
            {
                if (!keyframe || keyframe->isBad())
                {
                    continue;
                }
                if (layers[1])
                {
                    std::size_t group = map == active_map ? 2 : 8 + keyframe->mnOriginMapId % 6;
                    if (!keyframe->GetParent())
                    {
                        group = 3;
                    }
                    else if (layers[4] && map == active_map)
                    {
                        if (optimized_keyframes.contains(keyframe->mnId))
                        {
                            group = 4;
                        }
                        else if (fixed_keyframes.contains(keyframe->mnId))
                        {
                            group = 3;
                        }
                    }
                    add_camera(group, keyframe->GetPoseInverse(), _settings.key_frame_size);
                }
                if (map != active_map)
                {
                    continue;
                }
                const auto center = keyframe->GetCameraCenter();
                if (layers[2])
                {
                    for (auto *neighbor : keyframe->GetCovisiblesByWeight(100))
                    {
                        if (neighbor && !neighbor->isBad() && neighbor->mnId >= keyframe->mnId)
                        {
                            add_line(5, center, neighbor->GetCameraCenter());
                        }
                    }
                    if (auto *parent = keyframe->GetParent(); parent && !parent->isBad())
                    {
                        add_line(5, center, parent->GetCameraCenter());
                    }
                    for (auto *neighbor : keyframe->GetLoopEdges())
                    {
                        if (neighbor && !neighbor->isBad() && neighbor->mnId >= keyframe->mnId)
                        {
                            add_line(5, center, neighbor->GetCameraCenter());
                        }
                    }
                }
                if (layers[3] && active_map->isImuInitialized())
                {
                    if (auto *next = keyframe->mNextKF; next && !next->isBad())
                    {
                        add_line(6, center, next->GetCameraCenter());
                    }
                }
            }
        }
    }
    constexpr std::array<std::array<double, 3>, 14> colors{
        {{0.12, 0.15, 0.18},
         {0.9, 0.15, 0.12},
         {0.12, 0.35, 0.8},
         {0.9, 0.15, 0.12},
         {0.1, 0.7, 0.3},
         {0.1, 0.65, 0.4},
         {0.9, 0.25, 0.15},
         {0.0, 0.7, 0.3},
         {0.1, 0.4, 0.8},
         {0.7, 0.3, 0.8},
         {0.9, 0.2, 0.4},
         {0.5, 0.2, 0.8},
         {0.8, 0.7, 0.0},
         {0.0, 0.7, 0.8}}
    };
    for (std::size_t group = 0; group < points.size(); ++group)
    {
        if (points[group]->GetNumberOfPoints() == 0)
        {
            continue;
        }
        vtkNew<vtkPolyData> data;
        data->SetPoints(points[group]);
        if (group < 2)
        {
            data->SetVerts(cells[group]);
        }
        else
        {
            data->SetLines(cells[group]);
        }
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(data);
        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(colors[group][0], colors[group][1], colors[group][2]);
        actor->GetProperty()->SetPointSize(_settings.point_size);
        actor->GetProperty()->SetLineWidth(group == 7 ? _settings.camera_line_width : (group == 5 || group == 6 ? _settings.graph_line_width : _settings.key_frame_line_width));
        _renderer->AddActor(actor);
    }
    _renderer->ResetCameraClippingRange();
    _window->Render();
    _capture->Modified();
    _capture->Update();
    auto       *image      = _capture->GetOutput();
    const auto *dimensions = image->GetDimensions();
    return cv::Mat(dimensions[1], dimensions[0], CV_8UC3, image->GetScalarPointer()).clone();
}

void MapDrawer::release()
{
    _capture = nullptr;
    if (_window)
    {
        _window->Finalize();
    }
    _window   = nullptr;
    _renderer = nullptr;
}
} // namespace ORB_SLAM3