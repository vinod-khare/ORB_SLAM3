#pragma once

#include <array>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>

namespace ORB_SLAM3
{

class keypoint : public cv::KeyPoint
{
  public:
    keypoint() = default;
    keypoint(const cv::Point2f &position, float size, float angle = -1.f, float response = 0.f, int octave = 0, int class_id = -1)
        : cv::KeyPoint(position, size, angle, response, octave, class_id), display(position, size, angle, response, octave, class_id)
    {
    }

    keypoint(float x, float y, float size, float angle = -1.f, float response = 0.f, int octave = 0, int class_id = -1) : keypoint(cv::Point2f(x, y), size, angle, response, octave, class_id)
    {
    }

    keypoint(const cv::KeyPoint &tracked_keypoint) : cv::KeyPoint(tracked_keypoint), display(tracked_keypoint) {}
    keypoint(const cv::KeyPoint &tracked_keypoint, const cv::KeyPoint &display_keypoint) : cv::KeyPoint(tracked_keypoint), display(display_keypoint) {}

    cv::KeyPoint                 display;
    cv::Mat                      descriptor;
    cv::Mat                      place_descriptor;
    std::array<unsigned char, 3> color{255, 255, 255};
    bool                         has_color{false};
};

inline std::vector<cv::KeyPoint> to_cv_keypoints(const std::vector<keypoint> &keypoints)
{
    std::vector<cv::KeyPoint> converted;
    converted.reserve(keypoints.size());
    for (const auto &item : keypoints)
    {
        converted.push_back(static_cast<const cv::KeyPoint &>(item));
    }
    return converted;
}

inline cv::Mat descriptor_matrix(const std::vector<keypoint> &keypoints, std::size_t offset = 0)
{
    cv::Mat descriptors;
    for (std::size_t index = offset; index < keypoints.size(); ++index)
    {
        if (!keypoints[index].descriptor.empty())
        {
            descriptors.push_back(keypoints[index].descriptor);
        }
    }
    return descriptors;
}

inline std::vector<cv::Mat> place_descriptor_vector(const std::vector<keypoint> &keypoints)
{
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(keypoints.size());
    for (const auto &item : keypoints)
    {
        if (!item.place_descriptor.empty())
        {
            descriptors.push_back(item.place_descriptor);
        }
    }
    return descriptors;
}

} // namespace ORB_SLAM3
