#include "frame_prefetcher.h"

#include "Frame.h"
#include "time_this.h"

#include <chrono>
#include <optional>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace
{
void preprocess_image(cv::Mat &image, const cv::Size &size, cv::CLAHE &clahe, bool resize, cv::Mat *color = nullptr)
{
    if (resize)
    {
        cv::resize(image, image, size);
    }
    if (color)
    {
        *color = image.clone();
    }
    if (image.channels() == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    clahe.apply(image, image);
}
} // namespace

frame_prefetcher::frame_prefetcher(folder_reader &reader, bool stereo, bool inertial, float image_scale, bool precompute_optical_flow)
    : _reader(reader), _stereo(stereo), _inertial(inertial), _image_scale(image_scale), _precompute_optical_flow(precompute_optical_flow),
      _worker([this](std::stop_token stop_token) { run(stop_token); })
{
}

frame_prefetcher::~frame_prefetcher() = default;

frame_prefetcher::frame frame_prefetcher::read()
{
    std::unique_lock<std::mutex> lock(_mutex);
    _ready.wait(lock, [this] { return !_frames.empty() || _finished; });

    if (_frames.empty())
    {
        if (_error)
        {
            std::rethrow_exception(_error);
        }
        throw std::out_of_range("No more prefetched frames");
    }

    frame result = std::move(_frames.front());

    _frames.pop_front();
    lock.unlock();
    _ready.notify_all();

    return result;
}

void frame_prefetcher::run(std::stop_token stop_token)
{
    try
    {
        cv::Ptr<cv::CLAHE>      clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        std::deque<cv::Mat>     right_images;
        std::optional<cv::Size> right_size;
        std::exception_ptr      right_error;
        bool                    right_finished = false;
        std::jthread            right_worker;
        if (_stereo)
        {
            right_worker = std::jthread(
                [this, &right_images, &right_size, &right_error, &right_finished](std::stop_token right_stop)
                {
                    try
                    {
                        const auto right_clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                        for (std::size_t index = 0; index < _reader.size(); ++index)
                        {
                            {
                                std::unique_lock<std::mutex> lock(_mutex);
                                if (!_ready.wait(lock, right_stop, [&right_images] { return right_images.empty(); }))
                                {
                                    return;
                                }
                            }

                            const auto &filename = _reader.right_image_path(index);
                            cv::Mat     image    = cv::imread(filename, cv::IMREAD_COLOR);
                            if (image.empty())
                            {
                                throw std::runtime_error("Failed to load image at: " + filename);
                            }
                            cv::Size size;
                            {
                                std::unique_lock<std::mutex> lock(_mutex);
                                if (!_ready.wait(lock, right_stop, [&right_size] { return right_size.has_value(); }))
                                {
                                    return;
                                }
                                size = *right_size;
                                right_size.reset();
                            }
                            preprocess_image(image, size, *right_clahe, _image_scale != 1.f);

                            {
                                std::lock_guard<std::mutex> lock(_mutex);
                                if (right_stop.stop_requested())
                                {
                                    return;
                                }
                                right_images.push_back(std::move(image));
                            }
                            _ready.notify_all();
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        right_error = std::current_exception();
                    }
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        right_finished = true;
                    }
                    _ready.notify_all();
                });
        }

        for (std::size_t index = 0; index < _reader.size(); ++index)
        {
            {
                std::unique_lock<std::mutex> lock(_mutex);
                if (!_ready.wait(lock, stop_token, [this] { return _frames.size() < _capacity; }))
                {
                    return;
                }
            }

            frame                               prepared;
            std::chrono::steady_clock::duration preprocessing_time{};
            {
                const ORB_SLAM3::time_this timer(preprocessing_time);
                prepared.filename = _reader.image_path(index);
                if (_inertial)
                {
                    auto input         = _reader.read<orbslam3::frame_mono_inertial>();
                    prepared.image     = std::move(input.image);
                    prepared.timestamp = input.timestamp;
                    prepared.imu       = std::move(input.imu);
                }
                else
                {
                    auto input         = _reader.read<orbslam3::frame_mono>();
                    prepared.image     = std::move(input.image);
                    prepared.timestamp = input.timestamp;
                }

                if (prepared.image.empty())
                {
                    throw std::runtime_error("Failed to load image at: " + prepared.filename);
                }

                cv::Size size;
                if (_image_scale != 1.f)
                {
                    size = cv::Size(static_cast<int>(prepared.image.cols * _image_scale), static_cast<int>(prepared.image.rows * _image_scale));
                }

                if (_stereo)
                {
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        right_size = size;
                    }
                    _ready.notify_all();
                }

                preprocess_image(prepared.image, size, *clahe, _image_scale != 1.f, _stereo ? nullptr : &prepared.color);
                if (_precompute_optical_flow)
                {
                    prepared.optical_flow_pyramid = ORB_SLAM3::Frame::build_optical_flow_pyramid(prepared.image);
                }

                if (_stereo)
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    if (!_ready.wait(lock, stop_token, [&right_images, &right_finished] { return !right_images.empty() || right_finished; }))
                    {
                        return;
                    }
                    if (right_images.empty())
                    {
                        if (right_error)
                        {
                            std::rethrow_exception(right_error);
                        }
                        throw std::runtime_error("Right image stream ended before the left image stream");
                    }
                    prepared.image_right = std::move(right_images.front());

                    right_images.pop_front();
                    
                    lock.unlock();

                    _ready.notify_all();
                }
            }

            prepared.preprocessing_ms = std::chrono::duration<double, std::milli>(preprocessing_time).count();

            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (stop_token.stop_requested())
                {
                    return;
                }
                _frames.push_back(std::move(prepared));
            }

            _ready.notify_all();
        }
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _error = std::current_exception();
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _finished = true;
    }

    _ready.notify_all();
}