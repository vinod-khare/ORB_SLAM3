#include "viewer_statistics.h"

#include <atomic>
#include <thread>

#include <Eigen/Geometry>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEST_CASE("Viewer history retains every processed frame in a bounded ring", "[viewer_statistics]")
{
    ORB_SLAM3::viewer_statistics statistics;
    REQUIRE(statistics.get_snapshot().count == 0);

    for (std::size_t index = 0; index < ORB_SLAM3::viewer_statistics::history_capacity + 7; ++index)
    {
        ORB_SLAM3::viewer_statistics::sample sample;
        sample.processing.frame.frame_id    = index;
        sample.processing.frame.timestamp   = 1000.0 + static_cast<double>(index) * 0.125;
        sample.processing.frame.frontend_ms = static_cast<double>(index);
        sample.sequence_frame               = index + 1;
        if (index == 0)
        {
            sample.camera_to_world = Sophus::SE3f{};
        }
        statistics.record(sample);
    }

    const auto snapshot = statistics.get_snapshot();
    REQUIRE(snapshot.count == ORB_SLAM3::viewer_statistics::history_capacity);
    REQUIRE(snapshot.next == 7);
    REQUIRE(snapshot.latest.sequence_frame == ORB_SLAM3::viewer_statistics::history_capacity + 7);
    REQUIRE_FALSE(snapshot.latest.camera_to_world.has_value());
    REQUIRE(snapshot.sequence_elapsed_seconds == Catch::Approx(306 * 0.125));
    for (std::size_t index = 0; index < snapshot.count; ++index)
    {
        const auto &frame = snapshot.history[(snapshot.next + index) % snapshot.count];
        REQUIRE(frame.frame_id == index + 7);
        REQUIRE(frame.frontend_ms == Catch::Approx(static_cast<double>(index + 7)));
    }
    statistics.record({});
    REQUIRE(snapshot.latest.sequence_frame == 307);
    REQUIRE(statistics.get_snapshot().latest.sequence_frame == 0);
}

TEST_CASE("Viewer snapshots remain coherent across playback and UI threads", "[viewer_statistics]")
{
    ORB_SLAM3::viewer_statistics statistics;
    std::atomic<bool>            done{false};
    std::jthread                 producer(
        [&]
        {
            for (std::size_t index = 1; index <= 1000; ++index)
            {
                ORB_SLAM3::viewer_statistics::sample sample;
                sample.sequence_frame            = index;
                sample.processing.frame.frame_id = index;
                sample.input_wait_ms             = static_cast<double>(index);
                statistics.record(sample);
            }
            done = true;
        });
    do
    {
        const auto snapshot = statistics.get_snapshot();
        REQUIRE(snapshot.latest.sequence_frame == snapshot.latest.processing.frame.frame_id);
        REQUIRE(snapshot.latest.input_wait_ms == static_cast<double>(snapshot.latest.sequence_frame));
        if (snapshot.count > 0)
        {
            const auto newest = (snapshot.next + ORB_SLAM3::viewer_statistics::history_capacity - 1) % ORB_SLAM3::viewer_statistics::history_capacity;
            REQUIRE(snapshot.history[newest].frame_id == snapshot.latest.sequence_frame);
        }
    } while (!done);
    REQUIRE(statistics.get_snapshot().latest.sequence_frame == 1000);
}

TEST_CASE("Viewer reports camera-to-world ZYX orientation in degrees", "[viewer_statistics]")
{
    const float sign = GENERATE(-1.f, 1.f);
    const auto  rotation =
        Eigen::AngleAxisf(sign * 0.7f, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(sign * 0.3f, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(sign * 0.2f, Eigen::Vector3f::UnitX());
    const Sophus::SE3f camera_to_world{
        rotation, Eigen::Vector3f{1.f, 2.f, 3.f}
    };
    const auto degrees = ORB_SLAM3::viewer_statistics::orientation_degrees(camera_to_world);
    REQUIRE(degrees.x() == Catch::Approx(sign * 11.459156f));
    REQUIRE(degrees.y() == Catch::Approx(sign * 17.188734f));
    REQUIRE(degrees.z() == Catch::Approx(sign * 40.107046f));
    REQUIRE(ORB_SLAM3::viewer_statistics::orientation_degrees(Sophus::SE3f{}).isZero());
}

TEST_CASE("Viewer formats durations and source timestamps without local timezone assumptions", "[viewer_statistics]")
{
    using ORB_SLAM3::viewer_statistics;
    REQUIRE(viewer_statistics::format_duration(0) == "00:00:00.000");
    REQUIRE(viewer_statistics::format_duration(3723.125) == "01:02:03.125");
    REQUIRE(viewer_statistics::format_duration(90061.125) == "25:01:01.125");
    REQUIRE(viewer_statistics::format_duration(-1.125) == "-00:00:01.125");
    REQUIRE(viewer_statistics::format_timestamp(0) == "1970-01-01 00:00:00.000 UTC");
    REQUIRE(viewer_statistics::format_timestamp(1520531829.125) == "2018-03-08 17:57:09.125 UTC");
    REQUIRE(viewer_statistics::format_timestamp(-0.125) == "1969-12-31 23:59:59.875 UTC");
}
