#include "folder_reader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>

namespace
{
} // namespace

folder_reader::folder_reader(const settings &reader_settings)
{
    if (reader_settings.frames_skip < 0 || reader_settings.frames_stride <= 0 || reader_settings.frames_take < 0)
    {
        throw std::invalid_argument("Frame skip and take must be non-negative, and frame stride must be positive");
    }

    const std::string       &left_image_path  = reader_settings.left_image_path;
    const std::string       &right_image_path = reader_settings.right_image_path;
    int                      frames_skip      = reader_settings.frames_skip;
    const int                frames_stride    = reader_settings.frames_stride;
    const int                frames_take      = reader_settings.frames_take;
    const std::string       &imu_path         = reader_settings.imu_path;

    std::vector<std::string> allImages;
    std::vector<double>      allTimestamps;
    allTimestamps.reserve(5000);
    allImages.reserve(5000);

    std::vector<std::pair<double, std::string>> parsed;
    for (const auto &entry : std::filesystem::directory_iterator(left_image_path))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        const std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg")
        {
            continue;
        }

        const std::string stem = entry.path().stem().string();
        if (!is_numeric_stem(stem))
        {
            throw std::runtime_error("Invalid image filename for timestamp inference: " + entry.path().string() +
                                     ". Expected numeric nanosecond timestamp in the filename stem.");
        }

        const double timestamp_seconds = std::stod(stem) / 1e9;
        parsed.emplace_back(timestamp_seconds, entry.path().string());
    }

    std::sort(parsed.begin(), parsed.end(), [](const auto &left, const auto &right) { return left.first < right.first; });

    for (const auto &[timestamp_seconds, image_path] : parsed)
    {
        allTimestamps.push_back(timestamp_seconds);
        allImages.push_back(image_path);
    }

    if (frames_skip > static_cast<int>(allImages.size()))
    {
        frames_skip = static_cast<int>(allImages.size());
    }

    for (int i = frames_skip, count = 0; i < static_cast<int>(allImages.size()); i += frames_stride)
    {
        if (frames_take > 0 && count >= frames_take)
        {
            break;
        }

        _images.push_back(allImages[i]);
        _time_stamps.push_back(allTimestamps[i]);
        count++;
    }

    if (!right_image_path.empty())
    {
        // Right images are paired with left images by identical filename.
        _right_images.reserve(_images.size());
        for (const auto &left : _images)
        {
            const std::filesystem::path right = std::filesystem::path(right_image_path) / std::filesystem::path(left).filename();
            if (!std::filesystem::is_regular_file(right))
            {
                throw std::runtime_error("Right stereo image not found for " + left + ": " + right.string());
            }
            _right_images.push_back(right.string());
        }

        spdlog::info("[folder_reader] Paired {} right stereo images from {}", _right_images.size(), right_image_path);
    }

    if (!imu_path.empty())
    {
        std::ifstream imu_file(imu_path);
        if (!imu_file.is_open())
        {
            throw std::runtime_error("Failed to open IMU CSV file: " + imu_path);
        }

        std::string line;
        size_t      line_number = 0;
        while (std::getline(imu_file, line))
        {
            ++line_number;
            line = trim(line);
            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            std::istringstream row(line);
            std::string        value;
            double             data[7];
            size_t             column = 0;
            while (std::getline(row, value, ',') && column < 7)
            {
                data[column++] = std::stod(trim(value));
            }

            if (column != 7 || std::getline(row, value, ','))
            {
                throw std::runtime_error("Invalid IMU CSV row at " + imu_path + ":" + std::to_string(line_number));
            }

            const double timestamp = data[0] / 1e9;
            if (!_imu_measurements.empty() && timestamp < _imu_measurements.back().t)
            {
                throw std::runtime_error("IMU timestamps are not ordered at " + imu_path + ":" + std::to_string(line_number));
            }

            _imu_measurements.emplace_back(data[4], data[5], data[6], data[1], data[2], data[3], timestamp);
        }

        if (_imu_measurements.empty())
        {
            throw std::runtime_error("IMU CSV file contains no measurements: " + imu_path);
        }

        if (!_time_stamps.empty())
        {
            const auto first_after_frame = std::upper_bound(_imu_measurements.begin(), _imu_measurements.end(), _time_stamps.front(),
                                                            [](double timestamp, const ORB_SLAM3::IMU::Point &measurement) { return timestamp < measurement.t; });
            if (first_after_frame != _imu_measurements.begin())
            {
                _imu_index = static_cast<size_t>(std::distance(_imu_measurements.begin(), first_after_frame) - 1);
            }
        }

        spdlog::info("[folder_reader] Loaded {} IMU measurements from {}", _imu_measurements.size(), imu_path);
    }

    spdlog::info("✅ [folder_reader] Final loaded frames: {} (skip={}, stride={}, take={})", _images.size(), frames_skip, frames_stride, frames_take);
}

size_t                             folder_reader::size() const { return _images.size(); }

const std::string                 &folder_reader::image_path(size_t idx) const { return _images.at(idx); }

double                             folder_reader::timestamp(size_t idx) const { return _time_stamps.at(idx); }

cv::Mat                            folder_reader::read_image(size_t idx) const { return cv::imread(_images.at(idx), cv::IMREAD_COLOR); }

std::vector<ORB_SLAM3::IMU::Point> folder_reader::collect_imu(double timestamp, bool is_first_frame) const
{
    std::vector<ORB_SLAM3::IMU::Point> imu;
    if (!is_first_frame)
    {
        while (_imu_index < _imu_measurements.size() && _imu_measurements[_imu_index].t <= timestamp)
        {
            imu.push_back(_imu_measurements[_imu_index]);
            ++_imu_index;
        }
    }

    return imu;
}

bool folder_reader::is_numeric_stem(const std::string &s)
{
    if (s.empty())
    {
        return false;
    }

    bool seen_digit = false;
    bool seen_dot   = false;
    for (char c : s)
    {
        if (std::isdigit(static_cast<unsigned char>(c)))
        {
            seen_digit = true;
            continue;
        }

        if (c == '.' && !seen_dot)
        {
            seen_dot = true;
            continue;
        }

        return false;
    }

    return seen_digit;
}

std::string folder_reader::trim(const std::string &s)
{
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
    {
        return {};
    }

    const size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}
