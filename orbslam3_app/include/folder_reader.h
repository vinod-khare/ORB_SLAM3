#pragma once

#include "orbslam3/frame_mono.h"
#include "orbslam3/frame_mono_inertial.h"
#include "orbslam3/frame_stereo.h"
#include "orbslam3/frame_stereo_inertial.h"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

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
                  timestamps_type type = timestamps_type::auto_detect, const std::string &imu_path = {}, const std::string &right_image_path = {});

    static timestamps_type          parse_timestamps_type(const std::string &value);
    static std::string              trim(const std::string &s);

    size_t                          size() const;
    const std::string              &image_path(size_t idx) const;
    double                          timestamp(size_t idx) const;
    cv::Mat                         read_image(size_t idx) const;

    template <typename Frame> Frame read() const
    {
        constexpr bool stereo   = std::is_same_v<Frame, orbslam3::frame_stereo> || std::is_same_v<Frame, orbslam3::frame_stereo_inertial>;
        constexpr bool inertial = std::is_same_v<Frame, orbslam3::frame_mono_inertial> || std::is_same_v<Frame, orbslam3::frame_stereo_inertial>;
        
        static_assert(std::is_same_v<Frame, orbslam3::frame_mono> || stereo || inertial, "Unsupported folder_reader frame type");

        if constexpr (stereo)
        {
            if (_right_images.empty())
            {
                throw std::runtime_error("Cannot read a stereo frame without right images");
            }
        }

        if constexpr (inertial)
        {
            if (_imu_measurements.empty())
            {
                throw std::runtime_error("Cannot read an inertial frame without IMU measurements");
            }
        }

        if (_index >= _images.size())
        {
            throw std::out_of_range("No more frames available in folder_reader::read()");
        }

        Frame frame{};
        frame.timestamp = _time_stamps[_index];

        if constexpr (stereo)
        {
            frame.image_left  = cv::imread(_images[_index], cv::IMREAD_COLOR);
            frame.image_right = cv::imread(_right_images[_index], cv::IMREAD_COLOR);
        }
        else
        {
            frame.image = cv::imread(_images[_index], cv::IMREAD_COLOR);
        }

        if constexpr (inertial)
        {
            frame.imu = collect_imu(frame.timestamp, _index == 0);
        }

        ++_index;

        return frame;
    }

  private:
    static bool                        is_numeric_stem(const std::string &s);
    std::vector<ORB_SLAM3::IMU::Point> collect_imu(double timestamp, bool is_first_frame) const;

    std::vector<std::string>           _images;
    std::vector<std::string>           _right_images;
    std::vector<double>                _time_stamps;
    std::vector<ORB_SLAM3::IMU::Point> _imu_measurements;
    mutable size_t                     _index     = 0;
    mutable size_t                     _imu_index = 0;
};
