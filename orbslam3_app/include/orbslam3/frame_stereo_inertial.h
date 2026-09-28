#pragma once

#include "ImuTypes.h"
#include "orbslam3/frame_stereo.h"

#include <vector>

namespace orbslam3
{
class frame_stereo_inertial : public frame_stereo
{
  public:
    std::vector<ORB_SLAM3::IMU::Point> imu;
};
} // namespace orbslam3
