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

#include "viewer_control.h"
#include "viewer_settings.h"
#include "viewer_statistics.h"

#include <atomic>

namespace ORB_SLAM3
{
class FrameDrawer;
class MapDrawer;

class Viewer
{
  public:
    /** All dependencies, including settings, must outlive the viewer. */
    Viewer(viewer_control &controls, FrameDrawer &frame_drawer, MapDrawer &map_drawer, float tracked_image_scale, const viewer_settings &settings);

    void Run();
    void RequestFinish();
    void RequestStop();
    bool isFinished() const;
    bool isStopped() const;
    void Release();
    bool exit_requested() const;
    bool paused() const;
    void process_commands();

    /** Set immutable input metadata before starting the viewer thread. */
    void set_sequence(std::string description, std::size_t frames);
    void publish_statistics(const viewer_statistics::sample &frame);

    bool both{false};

  private:
    viewer_statistics      _statistics;
    std::string            _sequence_description;
    std::size_t            _sequence_frames{};
    viewer_control        &_controls;
    FrameDrawer           &_frame_drawer;
    MapDrawer             &_map_drawer;
    const viewer_settings &_settings;
    float                  _tracked_image_scale;
    std::atomic<bool>      _finish_requested{false};
    std::atomic<bool>      _finished{false};
    std::atomic<bool>      _stop_requested{false};
    std::atomic<bool>      _stopped{true};
    std::atomic<bool>      _exit_requested{false};
    std::atomic<bool>      _paused{false};
    std::atomic<bool>      _reset_requested{false};
    std::atomic<bool>      _localization_requested{false};
    bool                   _localization_active{false};
    std::atomic<int>       _dbow_version{2};
    std::atomic<bool>      _dbow_error{false};
};
} // namespace ORB_SLAM3