#pragma once

namespace ORB_SLAM3
{

enum sensor_type : int
{
    monocular     = 0,
    stereo        = 1,
    rgbd          = 2,
    imu_monocular = 3,
    imu_stereo    = 4,
    imu_rgbd      = 5,
};

} // namespace ORB_SLAM3
