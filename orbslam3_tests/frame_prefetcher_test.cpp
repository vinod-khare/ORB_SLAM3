#include "Frame.h"
#include "frame_prefetcher.h"

#include <chrono>
#include <filesystem>
#include <fstream>

#include <opencv2/imgproc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace
{
struct image_sequence
{
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("orbslam3_prefetch_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    explicit image_sequence(bool different_right_size = false)
    {
        std::filesystem::create_directories(directory / "left");
        std::filesystem::create_directories(directory / "right");
        for (int index = 0; index < 8; ++index)
        {
            const std::string filename = std::to_string(1000000000LL + index * 50000000LL) + ".png";
            cv::Mat           image(64, 64, CV_8UC3);
            cv::RNG           random(100 + index);
            random.fill(image, cv::RNG::UNIFORM, 0, 256);
            cv::Mat right_image;
            cv::flip(image, right_image, 1);
            if (different_right_size)
            {
                cv::resize(right_image, right_image, cv::Size(80, 48));
            }
            cv::imwrite((directory / "left" / filename).string(), image);
            cv::imwrite((directory / "right" / filename).string(), right_image);
        }
        std::ofstream imu(directory / "imu.csv");
        for (int index = 0; index < 41; ++index)
        {
            imu << 950000000LL + index * 10000000LL << ",1,2,3,4,5,6\n";
        }
    }

    ~image_sequence()
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    folder_reader::settings settings(bool stereo, bool inertial) const
    {
        folder_reader::settings result;
        result.left_image_path  = (directory / "left").string();
        result.right_image_path = stereo ? (directory / "right").string() : "";
        result.imu_path         = inertial ? (directory / "imu.csv").string() : "";
        return result;
    }
};
} // namespace

TEST_CASE("Folder reader rejects invalid frame sampling parameters")
{
    const image_sequence sequence;
    auto                 settings = sequence.settings(false, false);

    settings.frames_skip = -1;
    REQUIRE_THROWS_AS(folder_reader(settings), std::invalid_argument);

    settings.frames_skip   = 0;
    settings.frames_stride = 0;
    REQUIRE_THROWS_AS(folder_reader(settings), std::invalid_argument);

    settings.frames_stride = 1;
    settings.frames_take   = -1;
    REQUIRE_THROWS_AS(folder_reader(settings), std::invalid_argument);
}

TEST_CASE("Folder reader rejects malformed and non-finite IMU values")
{
    const image_sequence sequence;
    const auto           settings = sequence.settings(false, true);
    const auto           row = GENERATE("950000000x,1,2,3,4,5,6", "950000000,nan,2,3,4,5,6", "950000000,1,2,3,4,5,inf");
    std::ofstream        imu(sequence.directory / "imu.csv");
    imu << row << '\n';
    imu.close();

    REQUIRE_THROWS_AS(folder_reader(settings), std::runtime_error);
}

TEST_CASE("Folder reader reports image decoding failures")
{
    const bool           stereo = GENERATE(false, true);
    const image_sequence sequence;
    folder_reader        reader(sequence.settings(stereo, false));
    const std::string    filename = "1000000000.png";
    std::filesystem::remove(sequence.directory / (stereo ? "right" : "left") / filename);

    if (stereo)
    {
        REQUIRE_THROWS_AS(reader.read<orbslam3::frame_stereo>(), std::runtime_error);
    }
    else
    {
        REQUIRE_THROWS_AS(reader.read<orbslam3::frame_mono>(), std::runtime_error);
        REQUIRE_THROWS_AS(reader.read_image(0), std::runtime_error);
    }
}

TEST_CASE("Prefetch preserves frame order, preprocessing and IMU associations")
{
    const bool           stereo                  = GENERATE(false, true);
    const bool           inertial                = GENERATE(false, true);
    const float          image_scale             = GENERATE(1.f, 0.5f);
    const bool           sampled                 = GENERATE(false, true);
    const bool           different_right_size    = GENERATE(false, true);
    const bool           precompute_optical_flow = GENERATE(false, true);
    const image_sequence sequence(different_right_size);
    auto                 settings = sequence.settings(stereo, inertial);
    if (sampled)
    {
        settings.frames_skip   = 1;
        settings.frames_stride = 2;
        settings.frames_take   = 2;
    }
    folder_reader    reader(settings);
    folder_reader    reference(settings);
    frame_prefetcher prefetch(reader, stereo, inertial, image_scale, precompute_optical_flow);
    const auto       clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        const auto prepared = prefetch.read();
        REQUIRE(prepared.filename == reference.image_path(index));
        REQUIRE(prepared.timestamp == reference.timestamp(index));
        REQUIRE(prepared.preprocessing_ms > 0.0);

        cv::Mat expected_color = reference.read_image(index);
        if (image_scale != 1.f)
        {
            cv::resize(expected_color, expected_color, cv::Size(32, 32));
        }
        cv::Mat expected_gray;
        cv::cvtColor(expected_color, expected_gray, cv::COLOR_BGR2GRAY);
        clahe->apply(expected_gray, expected_gray);
        REQUIRE(cv::norm(prepared.image, expected_gray, cv::NORM_INF) == 0);
        if (precompute_optical_flow)
        {
            const auto expected_pyramid = ORB_SLAM3::Frame::build_optical_flow_pyramid(expected_gray);
            REQUIRE(prepared.optical_flow_pyramid.size() == expected_pyramid.size());
            for (std::size_t level = 0; level < expected_pyramid.size(); ++level)
            {
                REQUIRE(cv::norm(prepared.optical_flow_pyramid[level], expected_pyramid[level], cv::NORM_INF) == 0);
            }
            REQUIRE(prepared.optical_flow_pyramid.front().data != prepared.image.data);
            ORB_SLAM3::Frame tracking_frame;
            tracking_frame.optical_flow_pyramid = prepared.optical_flow_pyramid;
            REQUIRE(tracking_frame.get_optical_flow_pyramid(prepared.image).front().data == prepared.optical_flow_pyramid.front().data);
        }
        else
        {
            REQUIRE(prepared.optical_flow_pyramid.empty());
        }
        if (stereo)
        {
            REQUIRE(prepared.color.empty());
            cv::Mat expected_right = cv::imread(reference.right_image_path(index), cv::IMREAD_COLOR);
            if (image_scale != 1.f)
            {
                cv::resize(expected_right, expected_right, cv::Size(32, 32));
            }
            cv::cvtColor(expected_right, expected_right, cv::COLOR_BGR2GRAY);
            clahe->apply(expected_right, expected_right);
            REQUIRE(cv::norm(prepared.image_right, expected_right, cv::NORM_INF) == 0);
        }
        else
        {
            REQUIRE(prepared.image_right.empty());
            REQUIRE(cv::norm(prepared.color, expected_color, cv::NORM_INF) == 0);
        }

        if (inertial)
        {
            const auto expected_imu = stereo ? reference.read<orbslam3::frame_stereo_inertial>().imu : reference.read<orbslam3::frame_mono_inertial>().imu;
            REQUIRE(prepared.imu.size() == expected_imu.size());
            for (std::size_t imu_index = 0; imu_index < expected_imu.size(); ++imu_index)
            {
                REQUIRE(prepared.imu[imu_index].t == expected_imu[imu_index].t);
                REQUIRE(prepared.imu[imu_index].a == expected_imu[imu_index].a);
                REQUIRE(prepared.imu[imu_index].w == expected_imu[imu_index].w);
            }
        }
        else
        {
            REQUIRE(prepared.imu.empty());
        }
    }
    REQUIRE_THROWS_AS(prefetch.read(), std::out_of_range);
}

TEST_CASE("Prefetch propagates decoding errors after preceding frames")
{
    const bool           right_failure = GENERATE(false, true);
    const image_sequence sequence;
    folder_reader        reader(sequence.settings(true, false));
    std::filesystem::remove(sequence.directory / (right_failure ? "right" : "left") / "1100000000.png");
    frame_prefetcher prefetch(reader, true, false, 1.f);
    REQUIRE(prefetch.read().timestamp == 1.0);
    REQUIRE(prefetch.read().timestamp == 1.05);
    REQUIRE_THROWS_AS(prefetch.read(), std::runtime_error);
}

TEST_CASE("Prefetch propagates preprocessing errors and stops both workers")
{
    const bool           stereo = GENERATE(false, true);
    const image_sequence sequence;
    folder_reader        reader(sequence.settings(stereo, false));
    frame_prefetcher     prefetch(reader, stereo, false, 0.f);
    REQUIRE_THROWS_AS(prefetch.read(), cv::Exception);
}

TEST_CASE("Prefetch joins its worker when the consumer stops early")
{
    const bool           stereo = GENERATE(false, true);
    const image_sequence sequence;
    for (int iteration = 0; iteration < 16; ++iteration)
    {
        folder_reader    reader(sequence.settings(stereo, false));
        frame_prefetcher prefetch(reader, stereo, false, 1.f);
        REQUIRE(prefetch.read().timestamp == 1.0);
    }
}