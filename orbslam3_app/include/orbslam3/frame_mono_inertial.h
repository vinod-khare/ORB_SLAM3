#pragma once

#include "ImuTypes.h"
#include "orbslam3/frame_mono.h"

#include <vector>

namespace orbslam3
{
class frame_mono_inertial : public frame_mono
{
  public:
    std::vector<ORB_SLAM3::IMU::Point> imu;
};
} // namespace orbslam3