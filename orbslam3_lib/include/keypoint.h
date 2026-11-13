#pragma once

#include <array>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace ORB_SLAM3
{

/** Native detector metadata, with octave set to the SLAM image-pyramid level. */
class keypoint : public cv::KeyPoint
{
  public:
    keypoint() = default;
    keypoint(const cv::Point2f &position, float size, float angle = -1.f, float response = 0.f, int octave = 0, int class_id = -1)
        : cv::KeyPoint(position, size, angle, response, octave, class_id), pt_undistorted(position), tracking_angle(angle)
    {
    }

    keypoint(float x, float y, float size, float angle = -1.f, float response = 0.f, int octave = 0, int class_id = -1) : keypoint(cv::Point2f(x, y), size, angle, response, octave, class_id)
    {
    }

    keypoint(const cv::KeyPoint &native_keypoint) : keypoint(native_keypoint, native_keypoint.angle) {}
    keypoint(const cv::KeyPoint &native_keypoint, float tracking_angle) : cv::KeyPoint(native_keypoint), pt_undistorted(native_keypoint.pt), tracking_angle(tracking_angle) {}

    /** Geometry coordinates; equal to pt for rectified images and native fisheye observations. */
    cv::Point2f pt_undistorted;
    /** ORB orientation used for descriptor matching, independent of the detector's native angle. */
    float                        tracking_angle{-1.f};
    cv::Mat                      descriptor;
    std::array<unsigned char, 3> color{255, 255, 255};
    bool                         has_color{false};
};

inline std::vector<cv::KeyPoint> to_cv_keypoints(const std::vector<keypoint> &keypoints, bool undistorted = false)
{
    std::vector<cv::KeyPoint> converted;
    converted.reserve(keypoints.size());

    for (const auto &item : keypoints)
    {
        converted.push_back(item);
        if (undistorted)
        {
            converted.back().pt = item.pt_undistorted;
        }
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

inline std::vector<cv::Mat> descriptor_vector(const std::vector<keypoint> &keypoints)
{
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(keypoints.size());

    for (const auto &item : keypoints)
    {
        if (!item.descriptor.empty())
        {
            descriptors.push_back(item.descriptor);
        }
    }

    return descriptors;
}

} // namespace ORB_SLAM3
