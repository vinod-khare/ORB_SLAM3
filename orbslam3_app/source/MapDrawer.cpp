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

#include "Atlas.h"
#include "KeyFrame.h"
#include "MapPoint.h"

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkWindowToImageFilter.h>

#include <algorithm>
#include <cmath>
#include <set>

#include <vtkMatrix4x4.h>

namespace ORB_SLAM3
{
MapDrawer::MapDrawer(Atlas &atlas, const viewer_settings &settings) : _atlas(atlas), _settings(settings), _ground_truth(settings.ground_truth_path, settings.ground_truth_body_to_camera) {}
MapDrawer::~MapDrawer() = default;

void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &camera_pose, unsigned long map_id)
{
    std::scoped_lock lock(_camera_mutex);
    _camera_pose = camera_pose.inverse();
    _pending_camera_trajectory.emplace_back(_camera_pose.translation(), _connect_camera_trajectory && _trajectory_map_id == map_id);
    _trajectory_map_id         = map_id;
    _connect_camera_trajectory = true;
}

void MapDrawer::break_camera_trajectory()
{
    std::scoped_lock lock(_camera_mutex);
    _connect_camera_trajectory = false;
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

cv::Mat MapDrawer::render(int width, int height, const std::array<bool, 7> &layers)
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
        if (!_ground_truth.samples().empty())
        {
            vtkNew<vtkPoints>    reference_points;
            vtkNew<vtkCellArray> reference_lines;
            for (const auto &[timestamp, position] : _ground_truth.samples())
            {
                const auto index = reference_points->InsertNextPoint(position.x(), position.y(), position.z());
                if (index > 0)
                {
                    const vtkIdType segment[]{index - 1, index};
                    reference_lines->InsertNextCell(2, segment);
                }
            }
            vtkNew<vtkPolyData> reference_data;
            reference_data->SetPoints(reference_points);
            reference_data->SetLines(reference_lines);
            vtkNew<vtkPolyDataMapper> reference_mapper;
            reference_mapper->SetInputData(reference_data);
            _ground_truth_actor = vtkSmartPointer<vtkActor>::New();
            _ground_truth_actor->SetMapper(reference_mapper);
            _ground_truth_actor->GetProperty()->SetColor(0.95, 0.45, 0.05);
            _ground_truth_actor->GetProperty()->SetLineWidth(7.f);
            _ground_truth_actor->GetProperty()->LightingOff();
        }
        if (!_camera_trajectory_points)
        {
            _camera_trajectory_points = vtkSmartPointer<vtkPoints>::New();
            _camera_trajectory_lines  = vtkSmartPointer<vtkCellArray>::New();
        }
        vtkNew<vtkPolyData> trajectory_data;
        trajectory_data->SetPoints(_camera_trajectory_points);
        trajectory_data->SetLines(_camera_trajectory_lines);
        vtkNew<vtkPolyDataMapper> trajectory_mapper;
        trajectory_mapper->SetInputData(trajectory_data);
        trajectory_mapper->SetResolveCoincidentTopologyToPolygonOffset();
        trajectory_mapper->SetRelativeCoincidentTopologyLineOffsetParameters(0, -2);
        _camera_trajectory_actor = vtkSmartPointer<vtkActor>::New();
        _camera_trajectory_actor->SetMapper(trajectory_mapper);
        _camera_trajectory_actor->GetProperty()->SetColor(0.0, 0.45, 0.65);
        _camera_trajectory_actor->GetProperty()->SetLineWidth(3.f);
        _camera_trajectory_actor->GetProperty()->LightingOff();
        reset_view(false);
    }
    _window->SetSize(std::clamp(width, 1, 4096), std::clamp(height, 1, 4096));
    _renderer->RemoveAllViewProps();
    std::vector<std::pair<Eigen::Vector3f, bool>> pending_trajectory;
    {
        std::unique_lock lock(_camera_mutex, std::try_to_lock);
        if (lock.owns_lock())
        {
            pending_trajectory.swap(_pending_camera_trajectory);
        }
    }
    for (const auto &[position, connected] : pending_trajectory)
    {
        const auto index = _camera_trajectory_points->InsertNextPoint(position.x(), position.y(), position.z());
        if (connected && index > 0)
        {
            const vtkIdType segment[]{index - 1, index};
            _camera_trajectory_lines->InsertNextCell(2, segment);
        }
    }
    if (!pending_trajectory.empty())
    {
        _camera_trajectory_points->Modified();
        _camera_trajectory_lines->Modified();
    }
    std::array<vtkSmartPointer<vtkPoints>, 14>           points;
    std::array<vtkSmartPointer<vtkCellArray>, 14>        cells;
    std::array<vtkSmartPointer<vtkUnsignedCharArray>, 2> point_colors;
    for (auto &colors : point_colors)
    {
        colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetNumberOfComponents(3);
        colors->SetName("RGB");
    }
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
    trajectory_samples estimated_trajectory;
    auto              *active_map = _atlas.GetCurrentMap();
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
                const auto color = point->color();
                point_colors[group]->InsertNextTypedTuple(color.data());
            }
        }
        for (auto *map : _atlas.GetAllMaps())
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
                if (!_ground_truth_alignment && _ground_truth_actor)
                {
                    estimated_trajectory.emplace_back(keyframe->mTimeStamp, center.cast<double>());
                }
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
    if (!_ground_truth_alignment && _ground_truth_actor)
    {
        _ground_truth_alignment = _ground_truth.alignment(estimated_trajectory, _settings.ground_truth_correct_scale);
    }
    if (layers[5] && _ground_truth_actor && _ground_truth_alignment)
    {
        if (!_ground_truth_actor->GetUserMatrix())
        {
            vtkNew<vtkMatrix4x4> transform;
            for (int row = 0; row < 4; ++row)
            {
                for (int column = 0; column < 4; ++column)
                {
                    transform->SetElement(row, column, (*_ground_truth_alignment)(row, column));
                }
            }
            _ground_truth_actor->SetUserMatrix(transform);
        }
        _renderer->AddActor(_ground_truth_actor);
    }
    if (layers[6])
    {
        _renderer->AddActor(_camera_trajectory_actor);
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
            data->GetPointData()->SetScalars(point_colors[group]);
        }
        else
        {
            data->SetLines(cells[group]);
        }
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(data);
        if (group < 2)
        {
            mapper->SetColorModeToDirectScalars();
            mapper->SetScalarModeToUsePointData();
            mapper->ScalarVisibilityOn();
        }
        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->LightingOff();
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
    _camera_trajectory_actor = nullptr;
    _ground_truth_actor      = nullptr;
    _capture                 = nullptr;
    if (_window)
    {
        _window->Finalize();
    }
    _window   = nullptr;
    _renderer = nullptr;
}
} // namespace ORB_SLAM3