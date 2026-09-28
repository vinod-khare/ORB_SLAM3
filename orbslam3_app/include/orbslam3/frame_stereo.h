#pragma once

#include <opencv2/core/core.hpp>

namespace orbslam3
{
class frame_stereo
{
  public:
    double  timestamp;
    cv::Mat image_left;
    cv::Mat image_right;
};
} // namespace orbslam3
