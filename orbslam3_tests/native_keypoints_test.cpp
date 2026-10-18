#include "descriptor_matcher.h"
#include "keypoint_extractor.h"

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

TEST_CASE("Display keypoints retain detector metadata without changing tracking keypoints", "[native_keypoints]")
{
    using detector  = ORB_SLAM3::keypoint_detector_type;
    const auto type = GENERATE(detector::fast, detector::agast, detector::gftt, detector::orb, detector::brisk, detector::sift, detector::akaze);
    CAPTURE(static_cast<int>(type));
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 3, 20, 7, type);
    cv::Mat                       image(256, 256, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);
    std::vector<int>          overlap{100, 200};
    std::vector<cv::KeyPoint> tracking_keys;
    cv::Mat                   descriptors;
    const int                 mono_count  = extractor(image, cv::noArray(), tracking_keys, descriptors, overlap);
    const auto                native_keys = extractor.get_display_keypoints();
    REQUIRE_FALSE(tracking_keys.empty());
    REQUIRE(native_keys.size() == tracking_keys.size());
    REQUIRE(descriptors.rows == static_cast<int>(tracking_keys.size()));
    REQUIRE(descriptors.cols == 32);
    REQUIRE(mono_count > 0);
    REQUIRE(mono_count < static_cast<int>(tracking_keys.size()));
    REQUIRE(std::any_of(tracking_keys.begin(), tracking_keys.end(), [](const auto &key) { return key.octave > 0; }));

    auto       raw_detector = reference_detector(type);
    const auto scales       = extractor.get_scale_factors();
    for (int level = 0; level < extractor.get_levels(); ++level)
    {
        std::vector<cv::KeyPoint> raw_keys;
        raw_detector->detect(extractor.mvImagePyramid[level], raw_keys);
        for (std::size_t index = 0; index < tracking_keys.size(); ++index)
        {
            const auto &tracking_key = tracking_keys[index];
            if (tracking_key.octave != level)
            {
                continue;
            }
            const auto &native = native_keys[index];
            REQUIRE(native.pt == tracking_key.pt);
            REQUIRE(tracking_key.size == static_cast<int>(31 * scales[level]));
            REQUIRE(tracking_key.angle >= 0);
            REQUIRE(tracking_key.angle < 360);
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
    for (std::size_t index = 0; index < tracking_keys.size(); ++index)
    {
        REQUIRE(repeated_keys[index].pt == tracking_keys[index].pt);
        REQUIRE(repeated_keys[index].size == tracking_keys[index].size);
        REQUIRE(repeated_keys[index].angle == tracking_keys[index].angle);
    }
    extractor(cv::Mat::zeros(image.size(), CV_8UC1), cv::noArray(), repeated_keys, repeated_descriptors, overlap);
    REQUIRE(repeated_keys.empty());
    REQUIRE(extractor.get_display_keypoints().empty());
}

TEST_CASE("Keypoint extraction supports portrait images", "[native_keypoints]")
{
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 1, 20, 7);
    cv::Mat                       image(900, 320, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);

    std::vector<int>          overlap{0, 0};
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;

    extractor(image, cv::noArray(), keypoints, descriptors, overlap);

    REQUIRE_FALSE(keypoints.empty());
    REQUIRE(descriptors.rows == static_cast<int>(keypoints.size()));
    REQUIRE(descriptors.cols == 32);
}

TEST_CASE("OpenCV descriptors are selectable independently of the detector", "[native_keypoints][descriptors]")
{
    using namespace ORB_SLAM3;
    const auto detector   = GENERATE(keypoint_detector_type::fast, keypoint_detector_type::sift, keypoint_detector_type::brisk);
    const auto descriptor = GENERATE(keypoint_descriptor_type::orb, keypoint_descriptor_type::sift, keypoint_descriptor_type::brisk);
    CAPTURE(static_cast<int>(detector), static_cast<int>(descriptor));
    keypoint_extractor extractor(400, 1.2f, 3, 20, 7, detector, descriptor);
    cv::Mat            image(256, 256, CV_8UC1);
    cv::RNG            random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);
    std::vector<int>          overlap{100, 200};
    std::vector<cv::KeyPoint> keys;
    cv::Mat                   descriptors;
    extractor(image, cv::noArray(), keys, descriptors, overlap);
    REQUIRE_FALSE(keys.empty());
    REQUIRE(descriptors.rows == static_cast<int>(keys.size()));
    REQUIRE(extractor.get_place_descriptors().rows == descriptors.rows);
    REQUIRE(extractor.get_place_descriptors().type() == CV_8U);
    REQUIRE(extractor.get_place_descriptors().cols == 32);
    REQUIRE(descriptors.type() == (descriptor == keypoint_descriptor_type::sift ? CV_32F : CV_8U));
    REQUIRE(descriptors.cols == (descriptor == keypoint_descriptor_type::sift ? 128 : descriptor == keypoint_descriptor_type::brisk ? 64 : 32));
    REQUIRE(extractor.get_display_keypoints().size() == keys.size());
    std::vector<cv::KeyPoint> repeated_keys;
    cv::Mat                   repeated_descriptors;
    const auto                place_descriptors = extractor.get_place_descriptors().clone();
    extractor(image, cv::noArray(), repeated_keys, repeated_descriptors, overlap);
    REQUIRE(cv::norm(descriptors, repeated_descriptors, cv::NORM_INF) == 0);
    REQUIRE(cv::norm(place_descriptors, extractor.get_place_descriptors(), cv::NORM_INF) == 0);
    extractor(cv::Mat(), cv::noArray(), keys, descriptors, overlap);
    REQUIRE(keys.empty());
    REQUIRE(descriptors.empty());
    REQUIRE(extractor.get_place_descriptors().empty());
}

TEST_CASE("Keypoint extraction handles small pyramid levels", "[native_keypoints]")
{
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 2, 20, 7);
    cv::Mat                       image(70, 70, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);

    std::vector<int>          overlap{0, 0};
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;

    extractor(image, cv::noArray(), keypoints, descriptors, overlap);

    REQUIRE_FALSE(keypoints.empty());
    REQUIRE(descriptors.rows == static_cast<int>(keypoints.size()));
    REQUIRE(descriptors.cols == 32);
}

TEST_CASE("OpenCV matchers support binary and floating-point descriptors", "[descriptors][matching]")
{
    using namespace ORB_SLAM3;
    const auto backend    = GENERATE(descriptor_matcher_type::brute_force, descriptor_matcher_type::flann);
    const auto descriptor = GENERATE(keypoint_descriptor_type::orb, keypoint_descriptor_type::sift, keypoint_descriptor_type::brisk);
    CAPTURE(static_cast<int>(backend), static_cast<int>(descriptor));
    descriptor_matcher::set_backend(backend);
    struct reset_backend
    {
        ~reset_backend() { descriptor_matcher::set_backend(descriptor_matcher_type::brute_force); }
    } reset;
    const int columns = descriptor == keypoint_descriptor_type::sift ? 128 : descriptor == keypoint_descriptor_type::brisk ? 64 : 32;
    cv::Mat   train(100, columns, descriptor == keypoint_descriptor_type::sift ? CV_32F : CV_8U);
    cv::RNG   random(71);
    random.fill(train, cv::RNG::UNIFORM, 0, 256);
    const auto                           query = train.row(37).clone();
    std::vector<std::vector<cv::DMatch>> matches;
    descriptor_matcher::knn_match(query, train, matches);
    REQUIRE(matches.size() == 1);
    REQUIRE_FALSE(matches.front().empty());
    REQUIRE(matches.front().front().trainIdx == 37);
    REQUIRE(matches.front().front().distance == 0);
    REQUIRE(descriptor_matcher::descriptor_distance(query, train.row(37)) == 0);
    const auto ranked = descriptor_matcher::rank_candidates(query, train, {12, 37, 45});
    REQUIRE_FALSE(ranked.empty());
    REQUIRE(ranked.front() == 37);
    descriptor_matcher::knn_match(query, train, matches, train.rows);
    REQUIRE(matches.size() == 1);
    REQUIRE(matches.front().size() == static_cast<std::size_t>(train.rows));
    REQUIRE(matches.front().front().trainIdx == 37);
    descriptor_matcher::knn_match(query, train.row(37), matches);
    REQUIRE(matches.size() == 1);
    REQUIRE(matches.front().size() == 1);
    descriptor_matcher::knn_match(query, cv::Mat(), matches);
    REQUIRE(matches.empty());
    REQUIRE(descriptor_matcher::rank_candidates(query, train, {}).empty());
}

TEST_CASE("Descriptor distances retain precision and account for binary length", "[descriptors][matching]")
{
    using ORB_SLAM3::descriptor_matcher;
    REQUIRE(descriptor_matcher::descriptor_distance(cv::Mat::zeros(1, 32, CV_8U), cv::Mat(1, 32, CV_8U, cv::Scalar(255))) == 256);
    REQUIRE(descriptor_matcher::descriptor_distance(cv::Mat::zeros(1, 64, CV_8U), cv::Mat(1, 64, CV_8U, cv::Scalar(255))) == 256);
    REQUIRE(descriptor_matcher::descriptor_distance(cv::Mat::zeros(1, 128, CV_32F), cv::Mat(1, 128, CV_32F, cv::Scalar(0.25))) > 0);
}
