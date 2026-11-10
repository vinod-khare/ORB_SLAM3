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

#include "atlas.h"
#include "keypoint.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace ORB_SLAM3
{

class frame;
class tracking;

class frame_drawer
{
  public:
    explicit frame_drawer(atlas &atlas);

    // Update info from the last processed frame.
    bool update(tracking &tracker);

    // Draw last processed frame.
    cv::Mat draw_frame(float image_scale = 1.f);
    cv::Mat draw_right_frame(float image_scale = 1.f);

    // Smoothed rate at which frames are processed by the tracker.
    float get_tracking_fps();

    bool _both;

  protected:
    void draw_text_info(cv::Mat &image, int state, cv::Mat &image_with_info);

    // Info of the frame to be drawn
    std::shared_ptr<const frame> _current_frame;
    std::atomic<bool>            _localization_only{false};
    int                          _tracked_count{};
    int                          _tracked_odometry_count{};
    std::vector<keypoint>        _initial_keys;
    std::vector<int>             _initial_matches;
    int                          _state;

    atlas &_atlas;

    std::mutex                                       _mutex;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> _tracks;

    void update_tracking_fps();

    std::chrono::steady_clock::time_point _last_frame_time;
    bool                                  _has_last_frame_time = false;
    std::atomic<float>                    _tracking_fps{0.f};
    std::atomic<bool>                     _update_requested{true};
};

} // namespace ORB_SLAM3
