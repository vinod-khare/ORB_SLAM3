#pragma once

#include <chrono>

namespace ORB_SLAM3
{

/** Writes monotonic elapsed time to the supplied duration when the scope exits. */
class time_this
{
  public:
    time_this()                             = delete;
    time_this(const time_this &)            = delete;
    time_this(time_this &&)                 = delete;
    time_this &operator=(const time_this &) = delete;
    time_this &operator=(time_this &&)      = delete;

    explicit time_this(std::chrono::steady_clock::duration &time) : _time{time} {}

    ~time_this() { _time = std::chrono::steady_clock::now() - _start; }

  private:
    std::chrono::steady_clock::time_point _start{std::chrono::steady_clock::now()};
    std::chrono::steady_clock::duration  &_time;
};

} // namespace ORB_SLAM3