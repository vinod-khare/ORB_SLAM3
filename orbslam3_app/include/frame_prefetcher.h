#pragma once

#include "folder_reader.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stop_token>
#include <thread>

class frame_prefetcher
{
  public:
    struct frame
    {
        cv::Mat                            image;
        cv::Mat                            image_right;
        cv::Mat                            color;
        cv::Mat                            color_right;
        std::vector<cv::Mat>               optical_flow_pyramid;
        double                             timestamp{};
        double                             preprocessing_ms{};
        std::vector<ORB_SLAM3::IMU::Point> imu;
        std::string                        filename;
    };

    frame_prefetcher(folder_reader &reader, bool stereo, bool inertial, float image_scale, bool precompute_optical_flow = false);
    ~frame_prefetcher();

    frame read();

  private:
    void run(std::stop_token stop_token);

    folder_reader               &_reader;
    const bool                   _stereo;
    const bool                   _inertial;
    const float                  _image_scale;
    const bool                   _precompute_optical_flow;
    static constexpr std::size_t _capacity = 3;
    std::mutex                   _mutex;
    std::condition_variable_any  _ready;
    std::deque<frame>            _frames;
    std::exception_ptr           _error;
    bool                         _finished{false};
    std::jthread                 _worker;
};