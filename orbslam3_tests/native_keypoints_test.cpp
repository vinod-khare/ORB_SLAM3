#include "Frame.h"
#include "MapPoint.h"
#include "Pinhole.h"
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

TEST_CASE("Map point colors sample original pixels in RGB order", "[point_color]")
{
    ORB_SLAM3::Frame frame;
    frame.Nleft                                 = -1;
    frame.mvKeys                                = {cv::KeyPoint(1.f, 1.f, 1.f), cv::KeyPoint(-1.f, 0.f, 1.f), cv::KeyPoint(2.9f, 2.9f, 1.f)};
    frame.mvKeysUn                              = {cv::KeyPoint(0.f, 0.f, 1.f)};
    const int channels                          = GENERATE(1, 3, 4);
    frame.color_image                           = cv::Mat(3, 3, CV_MAKETYPE(CV_8U, channels), cv::Scalar(10, 20, 30, 40));
    const std::array<unsigned char, 3> expected = channels == 1 ? std::array<unsigned char, 3>{10, 10, 10} : std::array<unsigned char, 3>{30, 20, 10};
    REQUIRE(frame.keypoint_color(0) == expected);
    REQUIRE(frame.keypoint_color(2) == expected);
    const std::array<unsigned char, 3> white{255, 255, 255};
    REQUIRE(frame.keypoint_color(-1) == white);
    REQUIRE(frame.keypoint_color(1) == white);
    REQUIRE(frame.keypoint_color(3) == white);
    frame.color_is_rgb                     = true;
    const std::array<unsigned char, 3> rgb = channels == 1 ? expected : std::array<unsigned char, 3>{10, 20, 30};
    REQUIRE(frame.keypoint_color(0) == rgb);
    frame.Nleft             = 3;
    frame.mvKeysRight       = {cv::KeyPoint(0.f, 0.f, 1.f)};
    frame.color_image_right = cv::Mat(1, 1, CV_8UC1, cv::Scalar(70));
    const std::array<unsigned char, 3> gray{70, 70, 70};
    REQUIRE(frame.keypoint_color(3) == gray);
    frame.color_image.release();
    REQUIRE(frame.keypoint_color(0) == white);
}

TEST_CASE("Map points retain their first observation color", "[point_color]")
{
    cv::Mat image(128, 128, CV_8UC1);
    cv::RNG random(42);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::keypoint_extractor extractor(100, 1.2f, 1, 20, 7);
    ORB_SLAM3::Pinhole            camera({100, 100, 64, 64});
    cv::Mat                       distortion = cv::Mat::zeros(4, 1, CV_32F);
    ORB_SLAM3::Frame::mbInitialComputations  = true;
    ORB_SLAM3::Frame frame(image, 0.0, &extractor, nullptr, &camera, distortion, 10.f, 40.f);
    REQUIRE(frame.N >= 2);
    frame.SetPose(Sophus::SE3f{});
    frame.mnDataset   = 0;
    frame.color_image = cv::Mat(128, 128, CV_8UC3, cv::Scalar(10, 30, 200));
    ORB_SLAM3::Map map;
    const auto     second_keypoint                                                          = frame.mvKeys[1].pt;
    frame.color_image.at<cv::Vec3b>(cvRound(second_keypoint.y), cvRound(second_keypoint.x)) = cv::Vec3b(220, 40, 10);
    ORB_SLAM3::KeyFrame colored_keyframe(frame, &map, nullptr, &camera);
    REQUIRE(colored_keyframe.mvKeys[0].color == frame.keypoint_color(0));
    const std::array<unsigned char, 3> second_color{10, 40, 220};
    REQUIRE(colored_keyframe.mvKeys[1].color == second_color);
    ORB_SLAM3::MapPoint                point;
    const std::array<unsigned char, 3> white{255, 255, 255};
    REQUIRE(point.color() == white);
    point.AddObservation(&colored_keyframe, 0);
    REQUIRE(point.color() == colored_keyframe.mvKeys[0].color);
    point.AddObservation(&colored_keyframe, 1);
    REQUIRE(point.color() == colored_keyframe.mvKeys[0].color);
}

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
    std::vector<int>                 overlap{100, 200};
    std::vector<ORB_SLAM3::keypoint> tracking_keys;
    cv::Mat                          descriptors;
    const int                        mono_count = extractor(image, cv::noArray(), tracking_keys, descriptors, overlap);
    REQUIRE_FALSE(tracking_keys.empty());
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
            const auto &native = tracking_key.display;
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

    std::vector<ORB_SLAM3::keypoint> repeated_keys;
    cv::Mat                          repeated_descriptors;
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
}

TEST_CASE("Keypoint extraction supports portrait images", "[native_keypoints]")
{
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 1, 20, 7);
    cv::Mat                       image(900, 320, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);

    std::vector<int>                 overlap{0, 0};
    std::vector<ORB_SLAM3::keypoint> keypoints;
    cv::Mat                          descriptors;

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
    std::vector<int>      overlap{100, 200};
    std::vector<keypoint> keys;
    cv::Mat               descriptors;
    extractor(image, cv::noArray(), keys, descriptors, overlap);
    REQUIRE_FALSE(keys.empty());
    REQUIRE(descriptors.rows == static_cast<int>(keys.size()));
    REQUIRE(descriptors.type() == (descriptor == keypoint_descriptor_type::sift ? CV_32F : CV_8U));
    REQUIRE(descriptors.cols == (descriptor == keypoint_descriptor_type::sift ? 128 : descriptor == keypoint_descriptor_type::brisk ? 64 : 32));
    REQUIRE(cv::norm(descriptor_matrix(keys), descriptors, cv::NORM_INF) == 0);
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        REQUIRE(keys[index].descriptor.type() == descriptors.type());
        REQUIRE(keys[index].descriptor.cols == descriptors.cols);
        REQUIRE(keys[index].place_descriptor.type() == CV_8U);
        REQUIRE(keys[index].place_descriptor.cols == 32);
    }
    std::vector<keypoint> repeated_keys;
    cv::Mat               repeated_descriptors;
    extractor(image, cv::noArray(), repeated_keys, repeated_descriptors, overlap);
    REQUIRE(cv::norm(descriptors, repeated_descriptors, cv::NORM_INF) == 0);
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        REQUIRE(cv::norm(keys[index].place_descriptor, repeated_keys[index].place_descriptor, cv::NORM_INF) == 0);
    }
    extractor(cv::Mat(), cv::noArray(), keys, descriptors, overlap);
    REQUIRE(keys.empty());
    REQUIRE(descriptors.empty());
}

TEST_CASE("Keypoint extraction handles small pyramid levels", "[native_keypoints]")
{
    ORB_SLAM3::keypoint_extractor extractor(400, 1.2f, 2, 20, 7);
    cv::Mat                       image(70, 70, CV_8UC1);
    cv::RNG                       random(71);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.7);

    std::vector<int>                 overlap{0, 0};
    std::vector<ORB_SLAM3::keypoint> keypoints;
    cv::Mat                          descriptors;

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
