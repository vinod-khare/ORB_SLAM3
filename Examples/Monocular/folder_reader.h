#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

class folder_reader
{
public:
    folder_reader(const std::string &strImagePath, const std::string &strPathTimes,
                  int frames_skip = 0, int frames_stride = 1, int frames_take = 0);

    size_t size() const;
    const std::string& image_path(size_t idx) const;
    double timestamp(size_t idx) const;
    cv::Mat read_image(size_t idx) const;

private:
    static bool IsNumericStem(const std::string &s);

    std::vector<std::string> mImages;
    std::vector<double> mTimeStamps;
};
