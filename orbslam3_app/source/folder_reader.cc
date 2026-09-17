#include "folder_reader.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>

namespace
{

const char *timestamps_type_name(folder_reader::timestamps_type type)
{
    switch (type)
    {
    case folder_reader::timestamps_type::auto_detect:
        return "auto";
    case folder_reader::timestamps_type::filename_ns:
        return "filename_ns";
    case folder_reader::timestamps_type::timestamp_ns:
        return "timestamp_ns";
    case folder_reader::timestamps_type::utc:
        return "utc";
    }
    return "unknown";
}

std::vector<std::string> collect_sorted_image_paths(const std::string &image_dir)
{
    std::vector<std::string> image_paths;
    for (const auto &entry : std::filesystem::directory_iterator(image_dir))
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

        image_paths.push_back(entry.path().string());
    }

    std::sort(image_paths.begin(), image_paths.end());
    return image_paths;
}

// Build a map from stem to full path for fast lookups
std::unordered_map<std::string, std::string> build_stem_to_path_map(const std::string &image_dir)
{
    std::unordered_map<std::string, std::string> stem_map;
    for (const auto &entry : std::filesystem::directory_iterator(image_dir))
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

        stem_map[entry.path().stem().string()] = entry.path().string();
    }
    return stem_map;
}

bool parse_utc_timestamp_line(const std::string &line, double &timestamp_sec)
{
    std::string value = folder_reader::trim(line);
    if (value.size() < 19)
    {
        return false;
    }

    if (!(std::isdigit(static_cast<unsigned char>(value[0])) && std::isdigit(static_cast<unsigned char>(value[1])) && std::isdigit(static_cast<unsigned char>(value[2])) &&
          std::isdigit(static_cast<unsigned char>(value[3])) && value[4] == '-' && value[7] == '-' && (value[10] == ' ' || value[10] == 'T') && value[13] == ':' && value[16] == ':'))
    {
        return false;
    }

    std::tm time_info = {};
    time_info.tm_year = std::stoi(value.substr(0, 4)) - 1900;
    time_info.tm_mon  = std::stoi(value.substr(5, 2)) - 1;
    time_info.tm_mday = std::stoi(value.substr(8, 2));
    time_info.tm_hour = std::stoi(value.substr(11, 2));
    time_info.tm_min  = std::stoi(value.substr(14, 2));
    time_info.tm_sec  = std::stoi(value.substr(17, 2));

    double fractional = 0.0;
    size_t pos        = 19;
    if (pos < value.size() && value[pos] == '.')
    {
        size_t frac_end = pos + 1;
        while (frac_end < value.size() && std::isdigit(static_cast<unsigned char>(value[frac_end])))
        {
            ++frac_end;
        }

        const std::string frac_digits = value.substr(pos + 1, frac_end - pos - 1);
        if (!frac_digits.empty())
        {
            fractional = std::stod("0." + frac_digits);
        }

        pos = frac_end;
    }

    if (pos < value.size())
    {
        const std::string suffix = value.substr(pos);
        if (suffix != "Z" && suffix != " UTC")
        {
            return false;
        }
    }

    const time_t epoch = timegm(&time_info);
    timestamp_sec      = static_cast<double>(epoch) + fractional;
    return true;
}

bool file_exists_with_any_image_extension(const std::string &image_dir, const std::string &stem)
{
    static const char *exts[] = {".png", ".jpg", ".jpeg"};
    for (const char *ext : exts)
    {
        if (std::filesystem::exists(std::filesystem::path(image_dir) / (stem + ext)))
        {
            return true;
        }
    }

    return false;
}
} // namespace

folder_reader::folder_reader(const std::string &image_path, const std::string &times_path, int frames_skip, int frames_stride, int frames_take, timestamps_type type,
                             const std::string &imu_path)
{
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    std::vector<std::string> allImages;
    std::vector<double>      allTimestamps;
    allTimestamps.reserve(5000);
    allImages.reserve(5000);

    if (!times_path.empty())
    {
        std::ifstream fTimes(times_path.c_str());
        if (!fTimes.is_open())
        {
            throw std::runtime_error("Failed to open timestamps file: " + times_path);
        }

        std::vector<std::string> lines;
        std::string              s;
        while (std::getline(fTimes, s))
        {
            s = trim(s);
            if (s.empty() || s[0] == '#')
            {
                continue;
            }
            lines.push_back(s);
        }

        if (lines.empty())
        {
            throw std::runtime_error("Timestamps file is empty: " + times_path);
        }

        spdlog::info("📄 [folder_reader] Loaded timestamps file: {}", times_path);
        spdlog::info("🔢 [folder_reader] Parsed timestamp lines: {}", lines.size());

        timestamps_type detected_type = type;
        if (detected_type == timestamps_type::auto_detect)
        {
            const std::string &sample = lines.front();
            double             utc_ts = 0.0;
            if (parse_utc_timestamp_line(sample, utc_ts))
            {
                detected_type = timestamps_type::utc;
            }
            else
            {
                std::istringstream iss(sample);
                std::string        first;
                std::string        second;
                iss >> first >> second;

                if (is_numeric_stem(first))
                {
                    detected_type = file_exists_with_any_image_extension(image_path, first) ? timestamps_type::filename_ns : timestamps_type::timestamp_ns;
                }
                else if (!first.empty() && is_numeric_stem(second))
                {
                    detected_type = timestamps_type::timestamp_ns;
                }
                else
                {
                    throw std::runtime_error("Could not auto-detect timestamps file format for: " + times_path +
                                             ". Use --timestamps-type to specify one of: auto, filename_ns, timestamp_ns, utc");
                }
            }
        }

        spdlog::info("🧭 [folder_reader] Timestamp format: {}", timestamps_type_name(detected_type));

        if (detected_type == timestamps_type::filename_ns)
        {
            spdlog::info("⏳ [folder_reader] Building image map (this may take a moment)...");
            auto stem_map = build_stem_to_path_map(image_path);
            spdlog::info("✓ [folder_reader] Found {} images in directory", stem_map.size());

            spdlog::info("⏳ [folder_reader] Processing {} timestamp entries...", lines.size());
            for (size_t idx = 0; idx < lines.size(); ++idx)
            {
                const std::string &line = lines[idx];
                std::istringstream iss(line);
                std::string        item;
                iss >> item;

                auto it = stem_map.find(item);
                if (it == stem_map.end())
                {
                    throw std::runtime_error("Timestamp-named image not found for entry: " + item);
                }

                allImages.push_back(it->second);
                double t = stod(item);
                allTimestamps.push_back(t / 1e9);

                // Progress update every 500 images
                if ((idx + 1) % 500 == 0)
                {
                    spdlog::info("⏳ [folder_reader] Processed {}/{} entries...", idx + 1, lines.size());
                }
            }
        }
        else
        {
            const std::vector<std::string> sorted_images = collect_sorted_image_paths(image_path);
            if (sorted_images.size() < lines.size())
            {
                spdlog::error("❌ [folder_reader] Image/timestamp mismatch detected");
                spdlog::error("   ├─ Image directory : {}", image_path);
                spdlog::error("   ├─ Number of images: {}", sorted_images.size());
                spdlog::error("   ├─ Number of timestamps: {}", lines.size());
                spdlog::error("   └─ Timestamps file : {}", times_path);
                throw std::runtime_error("Not enough images in directory for timestamps file: " + image_path);
            }

            spdlog::info("🖼️  [folder_reader] Images found: {}", sorted_images.size());

            for (size_t i = 0; i < lines.size(); ++i)
            {
                const std::string &line          = lines[i];
                double             timestamp_sec = 0.0;

                if (detected_type == timestamps_type::utc)
                {
                    if (!parse_utc_timestamp_line(line, timestamp_sec))
                    {
                        throw std::runtime_error("Invalid UTC timestamp line: " + line);
                    }
                }
                else if (detected_type == timestamps_type::timestamp_ns)
                {
                    std::istringstream iss(line);
                    std::string        first;
                    std::string        second;
                    iss >> first >> second;

                    const std::string token = is_numeric_stem(first) ? first : second;
                    if (!is_numeric_stem(token))
                    {
                        throw std::runtime_error("Invalid numeric timestamp line: " + line);
                    }

                    timestamp_sec = stod(token) / 1e9;
                }
                else
                {
                    throw std::runtime_error("Unsupported timestamps type while reading file: " + times_path);
                }

                allImages.push_back(sorted_images[i]);
                allTimestamps.push_back(timestamp_sec);
            }
        }
    }
    else
    {
        std::vector<std::pair<double, std::string>> parsed;
        for (const auto &entry : std::filesystem::directory_iterator(image_path))
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
                                         ". Expected numeric stem (integer or floating point). Provide times.txt or rename files.");
            }

            const double t_ns = stod(stem);
            parsed.push_back({t_ns / 1e9, entry.path().string()});
        }

        std::sort(parsed.begin(), parsed.end(), [](const std::pair<double, std::string> &a, const std::pair<double, std::string> &b) { return a.first < b.first; });

        for (const auto &item : parsed)
        {
            allTimestamps.push_back(item.first);
            allImages.push_back(item.second);
        }
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

folder_reader::timestamps_type folder_reader::parse_timestamps_type(const std::string &value)
{
    if (value == "auto")
    {
        return timestamps_type::auto_detect;
    }
    if (value == "filename_ns")
    {
        return timestamps_type::filename_ns;
    }
    if (value == "timestamp_ns")
    {
        return timestamps_type::timestamp_ns;
    }
    if (value == "utc")
    {
        return timestamps_type::utc;
    }

    throw std::runtime_error("Invalid timestamps type: " + value + ". Expected one of: auto, filename_ns, timestamp_ns, utc");
}

size_t               folder_reader::size() const { return _images.size(); }

const std::string   &folder_reader::image_path(size_t idx) const { return _images.at(idx); }

double               folder_reader::timestamp(size_t idx) const { return _time_stamps.at(idx); }

cv::Mat              folder_reader::read_image(size_t idx) const { return cv::imread(_images.at(idx), cv::IMREAD_COLOR); }

orbslam3::frame_mono folder_reader::read() const
{
    if (_index >= _images.size())
    {
        throw std::out_of_range("No more frames available in folder_reader::read()");
    }

    orbslam3::frame_mono frame{.timestamp = _time_stamps[_index], .image = cv::imread(_images[_index], cv::IMREAD_COLOR)};

    ++_index;
    return frame;
}

orbslam3::frame_mono_inertial folder_reader::read_mono_inertial() const
{
    if (_imu_measurements.empty())
    {
        throw std::runtime_error("Cannot read a mono-inertial frame without IMU measurements");
    }

    const bool                       is_first_frame = _index == 0;
    orbslam3::frame_mono_inertial frame{read(), {}};

    if (!is_first_frame)
    {
        while (_imu_index < _imu_measurements.size() && _imu_measurements[_imu_index].t <= frame.timestamp)
        {
            frame.imu.push_back(_imu_measurements[_imu_index]);
            ++_imu_index;
        }
    }

    return frame;
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
