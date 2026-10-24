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

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include <atomic>
#include <chrono>
#include <mutex>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace ORB_SLAM3
{

class Atlas;
class Tracking;

class FrameDrawer
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit FrameDrawer(Atlas &atlas);

    // Update info from the last processed frame.
    bool    Update(Tracking *tracker);

    // Draw last processed frame.
    cv::Mat DrawFrame(float imageScale = 1.f);
    cv::Mat DrawRightFrame(float imageScale = 1.f);

    // Smoothed rate at which frames are processed by the tracker.
    float   get_tracking_fps();

    bool    both;

  protected:
    void                                   draw_text_info(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat                                mIm, mImRight;
    int                                    N;
    std::vector<cv::KeyPoint>              mvCurrentKeys, mvCurrentKeysRight;
    std::vector<bool>                      mvbMap, mvbVO;
    std::atomic<bool>                      mbOnlyTracking{false};
    int                                    mnTracked, mnTrackedVO;
    std::vector<cv::KeyPoint>              mvIniKeys;
    std::vector<int>                       mvIniMatches;
    int                                    mState;
    std::vector<float>                     mvCurrentDepth;
    float                                  mThDepth;

    Atlas                                 &_atlas;

    std::mutex                             mMutex;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> mvTracks;

    void                                   update_tracking_fps();

    std::chrono::steady_clock::time_point  _last_frame_time;
    bool                                   _has_last_frame_time = false;
    std::atomic<float>                     _tracking_fps{0.f};
    std::atomic<bool>                      _update_requested{true};
};

} // namespace ORB_SLAM3

#endif // FRAMEDRAWER_H
