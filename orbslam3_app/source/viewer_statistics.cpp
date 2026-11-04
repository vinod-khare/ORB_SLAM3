#include "viewer_statistics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>

#include <fmt/chrono.h>

namespace ORB_SLAM3
{

void viewer_statistics::record(const sample &frame)
{
    const std::lock_guard lock(_mutex);
    if (!_first_timestamp)
    {
        _first_timestamp = frame.processing.frame.timestamp;
    }
    _snapshot.sequence_elapsed_seconds = frame.processing.frame.timestamp - *_first_timestamp;
    _snapshot.history[_snapshot.next]  = frame.processing.frame;
    _snapshot.next                     = (_snapshot.next + 1) % history_capacity;
    _snapshot.count                    = std::min(_snapshot.count + 1, history_capacity);
    _snapshot.latest                   = frame;
}

viewer_statistics::snapshot viewer_statistics::get_snapshot() const
{
    const std::lock_guard lock(_mutex);
    return _snapshot;
}

Eigen::Vector3f viewer_statistics::orientation_degrees(const Sophus::SE3f &camera_to_world)
{
    const auto rotation = camera_to_world.rotationMatrix();
    return Eigen::Vector3f{std::atan2(rotation(2, 1), rotation(2, 2)), std::atan2(-rotation(2, 0), std::hypot(rotation(0, 0), rotation(1, 0))), std::atan2(rotation(1, 0), rotation(0, 0))} *
           (180.f / std::numbers::pi_v<float>);
}

std::string viewer_statistics::format_duration(double seconds)
{
    const std::chrono::hh_mm_ss time{std::chrono::floor<std::chrono::milliseconds>(std::chrono::duration<double>{seconds})};
    return fmt::format("{}{:02}:{:02}:{:02}.{:03}", time.is_negative() ? "-" : "", time.hours().count(), time.minutes().count(), time.seconds().count(), time.subseconds().count());
}

std::string viewer_statistics::format_timestamp(double seconds)
{
    const auto milliseconds  = std::chrono::floor<std::chrono::milliseconds>(std::chrono::duration<double>{seconds});
    const auto whole_seconds = std::chrono::floor<std::chrono::seconds>(milliseconds);
    return fmt::format("{:%Y-%m-%d %H:%M:%S}.{:03} UTC", fmt::gmtime(static_cast<std::time_t>(whole_seconds.count())), (milliseconds - whole_seconds).count());
}

} // namespace ORB_SLAM3
