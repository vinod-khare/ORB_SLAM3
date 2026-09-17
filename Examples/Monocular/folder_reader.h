#pragma once

#include "orbslam3/frame_mono.h"
#include "orbslam3/frame_mono_inertial.h"

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

class folder_reader
{
  public:
    enum class timestamps_type
    {
        auto_detect,
        filename_ns,
        timestamp_ns,
        utc
    };

    folder_reader(const std::string &image_path, const std::string &times_path, int frames_skip = 0, int frames_stride = 1, int frames_take = 0,
                  timestamps_type type = timestamps_type::auto_detect, const std::string &imu_path = {});

    static timestamps_type        parse_timestamps_type(const std::string &value);
    static std::string            trim(const std::string &s);

    size_t                        size() const;
    const std::string            &image_path(size_t idx) const;
    double                        timestamp(size_t idx) const;
    cv::Mat                       read_image(size_t idx) const;

    orbslam3::frame_mono          read() const;
    orbslam3::frame_mono_inertial read_mono_inertial() const;

  private:
    static bool                        is_numeric_stem(const std::string &s);

    std::vector<std::string>           _images;
    std::vector<double>                _time_stamps;
    std::vector<ORB_SLAM3::IMU::Point> _imu_measurements;
    mutable size_t                     _index     = 0;
    mutable size_t                     _imu_index = 0;
};
