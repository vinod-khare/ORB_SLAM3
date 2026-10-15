#include "frame_prefetcher.h"

#include <utility>

#include <opencv2/imgproc.hpp>

frame_prefetcher::frame_prefetcher(folder_reader &reader, bool stereo, bool inertial, float image_scale)
    : _reader(reader), _stereo(stereo), _inertial(inertial), _image_scale(image_scale), _worker([this](std::stop_token stop_token) { run(stop_token); })
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
        cv::Ptr<cv::CLAHE> clahe         = cv::createCLAHE(3.0, cv::Size(8, 8));
        const auto         to_gray_clahe = [&clahe](cv::Mat &image)
        {
            if (image.channels() == 3)
            {
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
            }
            clahe->apply(image, image);
        };

        for (std::size_t index = 0; index < _reader.size(); ++index)
        {
            {
                std::unique_lock<std::mutex> lock(_mutex);
                if (!_ready.wait(lock, stop_token, [this] { return _frames.size() < _capacity; }))
                {
                    return;
                }
            }

            frame prepared;
            prepared.filename = _reader.image_path(index);
            if (_stereo && _inertial)
            {
                auto input           = _reader.read<orbslam3::frame_stereo_inertial>();
                prepared.image       = std::move(input.image_left);
                prepared.image_right = std::move(input.image_right);
                prepared.timestamp   = input.timestamp;
                prepared.imu         = std::move(input.imu);
            }
            else if (_stereo)
            {
                auto input           = _reader.read<orbslam3::frame_stereo>();
                prepared.image       = std::move(input.image_left);
                prepared.image_right = std::move(input.image_right);
                prepared.timestamp   = input.timestamp;
            }
            else if (_inertial)
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

            if (prepared.image.empty() || (_stereo && prepared.image_right.empty()))
            {
                throw std::runtime_error("Failed to load image at: " + prepared.filename + (_stereo ? " (or its right pair)" : ""));
            }

            if (_image_scale != 1.f)
            {
                const cv::Size size(static_cast<int>(prepared.image.cols * _image_scale), static_cast<int>(prepared.image.rows * _image_scale));
                cv::resize(prepared.image, prepared.image, size);
                if (_stereo)
                {
                    cv::resize(prepared.image_right, prepared.image_right, size);
                }
            }

            if (!_stereo)
            {
                prepared.color = prepared.image.clone();
            }

            to_gray_clahe(prepared.image);

            if (_stereo)
            {
                to_gray_clahe(prepared.image_right);
            }

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