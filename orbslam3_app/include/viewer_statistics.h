#pragma once

#include "processing_statistics.h"

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>

#include <sophus/se3.hpp>

namespace ORB_SLAM3
{

/** Coherent, bounded telemetry shared by playback and the UI thread. */
class viewer_statistics
{
  public:
    static constexpr std::size_t history_capacity = 300;

    struct sample
    {
        processing_statistics::snapshot processing;
        std::optional<Sophus::SE3f>     camera_to_world;
        std::size_t                     sequence_frame{};
        double                          input_wait_ms{};
        double                          viewer_publication_ms{};
    };

    struct snapshot
    {
        sample                                                            latest;
        std::array<processing_statistics::frame_sample, history_capacity> history{};
        std::size_t                                                       count{};
        std::size_t                                                       next{};
        double                                                            sequence_elapsed_seconds{};
    };

    /** Publish once per processed frame, independently of UI refresh rate. */
    void     record(const sample &frame);
    snapshot get_snapshot() const;

    /** ZYX Euler convention: R = Rz(yaw) * Ry(pitch) * Rx(roll), in degrees. */
    static Eigen::Vector3f orientation_degrees(const Sophus::SE3f &camera_to_world);
    static std::string     format_duration(double seconds);
    /** Interpret the dataset timestamp as Unix seconds, explicitly in UTC. */
    static std::string format_timestamp(double seconds);

  private:
    mutable std::mutex    _mutex;
    snapshot              _snapshot;
    std::optional<double> _first_timestamp;
};

} // namespace ORB_SLAM3
