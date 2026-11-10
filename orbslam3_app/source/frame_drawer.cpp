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

#include "frame_drawer.h"

#include "Atlas.h"
#include "MapPoint.h"
#include "Tracking.h"

#include <iomanip>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

namespace ORB_SLAM3
{

frame_drawer::frame_drawer(Atlas &atlas) : _both(false), _atlas(atlas)
{
    auto initial_frame               = std::make_shared<Frame>();
    initial_frame->color_image       = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    initial_frame->color_image_right = initial_frame->color_image.clone();
    _current_frame                   = std::move(initial_frame);
    _state                           = Tracking::SYSTEM_NOT_READY;
}

cv::Mat frame_drawer::draw_frame(float image_scale)
{
    _update_requested = true;
    cv::Mat                                im;
    std::shared_ptr<const Frame>           frame;
    vector<keypoint>                       vIniKeys;
    vector<int>                            vMatches;
    vector<pair<cv::Point2f, cv::Point2f>> vTracks;
    int                                    state;

    cv::Scalar standardColor(0, 255, 0);
    cv::Scalar odometryColor(255, 0, 0);

    // Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(_mutex);
        state = _state;
        frame = _current_frame;
        if (_state == Tracking::SYSTEM_NOT_READY)
        {
            _state = Tracking::NO_IMAGES_YET;
        }

        if (_state == Tracking::NOT_INITIALIZED)
        {
            vIniKeys = _initial_keys;
            vMatches = _initial_matches;
            vTracks  = _tracks;
        }
    }
    const auto &current_keys = frame->mvKeys;
    im                       = frame->color_image.clone();

    if (image_scale != 1.f)
    {
        int imWidth  = im.cols / image_scale;
        int imHeight = im.rows / image_scale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if (im.channels() < 3) // this should be always true
    {
        cvtColor(im, im, cv::COLOR_GRAY2BGR);
    }

    // Draw
    if (state == Tracking::NOT_INITIALIZED)
    {
        for (unsigned int i = 0; i < vMatches.size(); i++)
        {
            if (vMatches[i] >= 0)
            {
                cv::Point2f pt1, pt2;
                if (image_scale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / image_scale;
                    pt2 = current_keys[vMatches[i]].pt / image_scale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = current_keys[vMatches[i]].pt;
                }
                cv::line(im, pt1, pt2, standardColor);
            }
        }
        for (vector<pair<cv::Point2f, cv::Point2f>>::iterator it = vTracks.begin(); it != vTracks.end(); it++)
        {
            cv::Point2f pt1, pt2;
            if (image_scale != 1.f)
            {
                pt1 = (*it).first / image_scale;
                pt2 = (*it).second / image_scale;
            }
            else
            {
                pt1 = (*it).first;
                pt2 = (*it).second;
            }
            cv::line(im, pt1, pt2, standardColor, 5);
        }
    }
    else if (state == Tracking::OK) // TRACKING
    {
        _tracked_count          = 0;
        _tracked_odometry_count = 0;
        vector<cv::KeyPoint> map_keys, vo_keys;
        int                  n = current_keys.size();
        for (int i = 0; i < n; i++)
        {
            auto *map_point = frame->mvpMapPoints[i];
            if (map_point && !frame->mvbOutlier[i])
            {
                auto key  = current_keys[i].display;
                key.pt   /= image_scale;
                key.size /= image_scale;

                // This is a match to a MapPoint in the map
                if (map_point->Observations() > 0)
                {
                    map_keys.push_back(key);
                    _tracked_count++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    vo_keys.push_back(key);
                    _tracked_odometry_count++;
                }
            }
        }
        const auto flags = cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
        cv::drawKeypoints(im, map_keys, im, standardColor, flags);
        cv::drawKeypoints(im, vo_keys, im, odometryColor, flags);
    }

    cv::Mat imWithInfo;
    draw_text_info(im, state, imWithInfo);

    return imWithInfo;
}

cv::Mat frame_drawer::draw_right_frame(float image_scale)
{
    _update_requested = true;
    cv::Mat                      im;
    std::shared_ptr<const Frame> frame;
    vector<keypoint>             vIniKeys;
    vector<int>                  vMatches;
    int                          state;

    // Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(_mutex);
        state = _state;
        frame = _current_frame;
        if (_state == Tracking::SYSTEM_NOT_READY)
        {
            _state = Tracking::NO_IMAGES_YET;
        }

        if (_state == Tracking::NOT_INITIALIZED)
        {
            vIniKeys = _initial_keys;
            vMatches = _initial_matches;
        }
    } // destroy scoped mutex -> release mutex
    const auto &current_keys   = frame->mvKeysRight;
    const auto  left_key_count = frame->mvKeys.size();
    im                         = frame->color_image_right.clone();

    if (image_scale != 1.f)
    {
        int imWidth  = im.cols / image_scale;
        int imHeight = im.rows / image_scale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if (im.channels() < 3) // this should be always true
    {
        cvtColor(im, im, cv::COLOR_GRAY2BGR);
    }

    // Draw
    if (state == Tracking::NOT_INITIALIZED) // INITIALIZING
    {
        for (unsigned int i = 0; i < vMatches.size(); i++)
        {
            if (vMatches[i] >= 0)
            {
                cv::Point2f pt1, pt2;
                if (image_scale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / image_scale;
                    pt2 = current_keys[vMatches[i]].pt / image_scale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = current_keys[vMatches[i]].pt;
                }

                cv::line(im, pt1, pt2, cv::Scalar(0, 255, 0));
            }
        }
    }
    else if (state == Tracking::OK) // TRACKING
    {
        _tracked_count          = 0;
        _tracked_odometry_count = 0;
        vector<cv::KeyPoint> map_keys, vo_keys;

        for (std::size_t i = 0; i < current_keys.size(); i++)
        {
            const auto index     = i + left_key_count;
            auto      *map_point = frame->mvpMapPoints[index];
            if (map_point && !frame->mvbOutlier[index])
            {
                auto key  = current_keys[i].display;
                key.pt   /= image_scale;
                key.size /= image_scale;

                // This is a match to a MapPoint in the map
                if (map_point->Observations() > 0)
                {
                    map_keys.push_back(key);
                    _tracked_count++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    vo_keys.push_back(key);
                    _tracked_odometry_count++;
                }
            }
        }
        const auto flags = cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
        cv::drawKeypoints(im, map_keys, im, cv::Scalar(0, 255, 0), flags);
        cv::drawKeypoints(im, vo_keys, im, cv::Scalar(255, 0, 0), flags);
    }

    cv::Mat imWithInfo;
    draw_text_info(im, state, imWithInfo);

    return imWithInfo;
}

void frame_drawer::draw_text_info(cv::Mat &image, int state, cv::Mat &image_with_info)
{
    stringstream s;
    if (state == Tracking::NO_IMAGES_YET)
    {
        s << " WAITING FOR IMAGES";
    }
    else if (state == Tracking::NOT_INITIALIZED)
    {
        s << " TRYING TO INITIALIZE ";
    }
    else if (state == Tracking::OK)
    {
        if (!_localization_only)
        {
            s << "SLAM MODE |  ";
        }
        else
        {
            s << "LOCALIZATION | ";
        }
        int nMaps = _atlas.CountMaps();
        int nKFs  = _atlas.KeyFramesInMap();
        int nMPs  = _atlas.MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << _tracked_count;
        if (_tracked_odometry_count > 0)
        {
            s << ", + VO matches: " << _tracked_odometry_count;
        }
    }
    else if (state == Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if (state == Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    stringstream s_fps;
    if (state != Tracking::SYSTEM_NOT_READY && state != Tracking::NO_IMAGES_YET)
    {
        s_fps << std::fixed << std::setprecision(1) << get_tracking_fps() << " FPS";
    }

    int      baseline = 0;
    cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    const int lineHeight   = textSize.height + 10;
    const int nLines       = s_fps.str().empty() ? 1 : 2;
    const int bannerHeight = nLines * lineHeight;

    image_with_info = cv::Mat(image.rows + bannerHeight, image.cols, image.type());
    image.copyTo(image_with_info.rowRange(0, image.rows).colRange(0, image.cols));
    image_with_info.rowRange(image.rows, image_with_info.rows) = cv::Mat::zeros(bannerHeight, image.cols, image.type());
    cv::putText(image_with_info, s.str(), cv::Point(5, image.rows + lineHeight - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
    if (nLines == 2)
    {
        cv::putText(image_with_info, s_fps.str(), cv::Point(5, image.rows + 2 * lineHeight - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
    }
}

float frame_drawer::get_tracking_fps() { return _tracking_fps.load(); }

void frame_drawer::update_tracking_fps()
{
    const auto now = std::chrono::steady_clock::now();

    if (_has_last_frame_time)
    {
        const float dt = std::chrono::duration<float>(now - _last_frame_time).count();
        if (dt > 0.f)
        {
            const float instant_fps = 1.f / dt;
            // Exponential smoothing to keep the displayed value readable.
            const float alpha        = 0.1f;
            const float previous_fps = _tracking_fps.load();
            _tracking_fps            = (previous_fps > 0.f) ? (alpha * instant_fps + (1.f - alpha) * previous_fps) : instant_fps;
        }
    }
    _last_frame_time     = now;
    _has_last_frame_time = true;
}

bool frame_drawer::update(Tracking &tracker)
{
    update_tracking_fps();

    unique_lock<mutex> lock(_mutex, std::try_to_lock);
    if (!lock.owns_lock() || !_update_requested.exchange(false))
    {
        return false;
    }
    _current_frame = std::make_shared<Frame>(tracker.mCurrentFrame);

    _localization_only = tracker.mbOnlyTracking;

    if (tracker.mLastProcessedState == Tracking::NOT_INITIALIZED)
    {
        _initial_keys    = tracker.mInitialFrame.mvKeys;
        _initial_matches = tracker.mvIniMatches;
    }
    _state = static_cast<int>(tracker.mLastProcessedState);
    return true;
}

} // namespace ORB_SLAM3
