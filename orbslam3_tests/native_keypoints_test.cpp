#include "ORBextractor.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace
{

cv::Ptr<cv::Feature2D> reference_detector(ORB_SLAM3::keypoint_detector_type type)
{
    using detector = ORB_SLAM3::keypoint_detector_type;
    switch (type)
    {
    case detector::fast:
        return cv::FastFeatureDetector::create(20);
    case detector::agast:
        return cv::AgastFeatureDetector::create(20);
    case detector::gftt:
        return cv::GFTTDetector::create(0, 0.01, 1.0, 3, false);
    case detector::orb:
        return cv::ORB::create(400, 1.2f, 1, 19, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    case detector::brisk:
        return cv::BRISK::create(20, 0);
    case detector::sift:
        return cv::SIFT::create();
    case detector::akaze:
        return cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 1);
    }
    FAIL("Unknown detector type");
    return {};
}

} // namespace

TEST_CASE("Display keypoints retain detector metadata without changing ORB keypoints", "[native_keypoints]")
{
    using detector  = ORB_SLAM3::keypoint_detector_type;
    const auto type = GENERATE(detector::fast, detector::agast, detector::gftt, detector::orb, detector::brisk, detector::sift, detector::akaze);
    CAPTURE(static_cast<int>(type));
    ORB_SLAM3::ORBextractor extractor(400, 1.2f, 3, 20, 7, type);
    cv::Mat                 image(256, 256, CV_8UC1);
    cv::RNG                 random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);
    std::vector<int>          overlap{100, 200};
    std::vector<cv::KeyPoint> orb_keys;
    cv::Mat                   descriptors;
    const int                 mono_count  = extractor(image, cv::noArray(), orb_keys, descriptors, overlap);
    const auto                native_keys = extractor.get_display_keypoints();
    REQUIRE_FALSE(orb_keys.empty());
    REQUIRE(native_keys.size() == orb_keys.size());
    REQUIRE(descriptors.rows == static_cast<int>(orb_keys.size()));
    REQUIRE(descriptors.cols == 32);
    REQUIRE(mono_count > 0);
    REQUIRE(mono_count < static_cast<int>(orb_keys.size()));

    auto       raw_detector = reference_detector(type);
    const auto scales       = extractor.GetScaleFactors();
    for (int level = 0; level < extractor.GetLevels(); ++level)
    {
        std::vector<cv::KeyPoint> raw_keys;
        raw_detector->detect(extractor.mvImagePyramid[level], raw_keys);
        for (std::size_t index = 0; index < orb_keys.size(); ++index)
        {
            const auto &orb_key = orb_keys[index];
            if (orb_key.octave != level)
            {
                continue;
            }
            const auto &native = native_keys[index];
            REQUIRE(native.pt == orb_key.pt);
            REQUIRE(orb_key.size == static_cast<int>(31 * scales[level]));
            REQUIRE(orb_key.angle >= 0);
            REQUIRE(orb_key.angle < 360);
            const bool in_overlap = native.pt.x >= overlap[0] && native.pt.x <= overlap[1];
            REQUIRE(in_overlap == (index >= static_cast<std::size_t>(mono_count)));
            if (type == detector::fast || type == detector::agast)
            {
                REQUIRE(native.size == 7 * scales[level]);
                REQUIRE(native.angle == -1);
            }
            else
            {
                REQUIRE(std::any_of(raw_keys.begin(), raw_keys.end(),
                                    [&](const auto &raw)
                                    {
                                        return cv::norm(raw.pt * scales[level] - native.pt) < 0.001 && std::abs(raw.size * scales[level] - native.size) < 0.001 && raw.angle == native.angle &&
                                               raw.response == native.response && raw.class_id == native.class_id;
                                    }));
            }
        }
    }

    std::vector<cv::KeyPoint> repeated_keys;
    cv::Mat                   repeated_descriptors;
    REQUIRE(extractor(image, cv::noArray(), repeated_keys, repeated_descriptors, overlap) == mono_count);
    REQUIRE(cv::norm(descriptors, repeated_descriptors, cv::NORM_INF) == 0);
    for (std::size_t index = 0; index < orb_keys.size(); ++index)
    {
        REQUIRE(repeated_keys[index].pt == orb_keys[index].pt);
        REQUIRE(repeated_keys[index].size == orb_keys[index].size);
        REQUIRE(repeated_keys[index].angle == orb_keys[index].angle);
    }
    extractor(cv::Mat::zeros(image.size(), CV_8UC1), cv::noArray(), repeated_keys, repeated_descriptors, overlap);
    REQUIRE(repeated_keys.empty());
    REQUIRE(extractor.get_display_keypoints().empty());
}
