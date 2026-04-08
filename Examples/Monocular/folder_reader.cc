#include "folder_reader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

folder_reader::folder_reader(const std::string &strImagePath, const std::string &strPathTimes,
                             int frames_skip, int frames_stride, int frames_take)
{
    std::vector<std::string> allImages;
    std::vector<double> allTimestamps;
    allTimestamps.reserve(5000);
    allImages.reserve(5000);

    if(!strPathTimes.empty())
    {
        std::ifstream fTimes;
        fTimes.open(strPathTimes.c_str());
        while(!fTimes.eof())
        {
            std::string s;
            getline(fTimes,s);

            if(!s.empty())
            {
                if (s[0] == '#')
                    continue;

                int pos = s.find(' ');
                std::string item = s.substr(0, pos);

                allImages.push_back(strImagePath + "/" + item + ".png");
                double t = stod(item);
                allTimestamps.push_back(t/1e9);
            }
        }
    }
    else
    {
        std::vector<std::pair<double, std::string>> parsed;
        for (const auto &entry : std::filesystem::directory_iterator(strImagePath))
        {
            if (!entry.is_regular_file())
                continue;

            const std::string ext = entry.path().extension().string();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg")
                continue;

            const std::string stem = entry.path().stem().string();
            if (!IsNumericStem(stem))
                throw std::runtime_error("Invalid image filename for timestamp inference: " + entry.path().string() +
                                         ". Expected numeric stem (integer or floating point). Provide times.txt or rename files.");

            const double t_ns = stod(stem);
            parsed.push_back({t_ns / 1e9, entry.path().string()});
        }

        std::sort(parsed.begin(), parsed.end(),
                 [](const std::pair<double, std::string> &a, const std::pair<double, std::string> &b) {
                     return a.first < b.first;
                 });

        for (const auto &item : parsed)
        {
            allTimestamps.push_back(item.first);
            allImages.push_back(item.second);
        }
    }

    for (int i = frames_skip, count = 0; i < static_cast<int>(allImages.size()); i += frames_stride)
    {
        if (frames_take > 0 && count >= frames_take)
            break;

        mImages.push_back(allImages[i]);
        mTimeStamps.push_back(allTimestamps[i]);
        count++;
    }
}

size_t folder_reader::size() const
{
    return mImages.size();
}

const std::string& folder_reader::image_path(size_t idx) const
{
    return mImages.at(idx);
}

double folder_reader::timestamp(size_t idx) const
{
    return mTimeStamps.at(idx);
}

cv::Mat folder_reader::read_image(size_t idx) const
{
    return cv::imread(mImages.at(idx), cv::IMREAD_GRAYSCALE);
}

bool folder_reader::IsNumericStem(const std::string &s)
{
    if (s.empty()) return false;

    bool seen_digit = false;
    bool seen_dot = false;
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
