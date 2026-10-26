#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace ORB_SLAM3
{

/** Thread-safe, bounded statistics storage, independent of logging and rendering. */
class processing_statistics
{
  public:
    using clock = std::chrono::steady_clock;

    enum class backend_stage
    {
        LOCAL_MAPPING,
        LOOP_CLOSING,
        GLOBAL_BUNDLE_ADJUSTMENT
    };

    struct frame_sample
    {
        std::uint64_t frame_id{};
        double        timestamp{};
        std::size_t   detected_keypoints{};
        std::size_t   matched_keypoints{};
        double        preprocessing_ms{};
        double        frontend_ms{};
    };

    struct backend_sample
    {
        std::uint64_t jobs{};
        double        last_ms{};
        double        total_ms{};
    };

    struct snapshot
    {
        frame_sample   frame;
        std::uint64_t  frames{};
        double         elapsed_seconds{};
        double         throughput_fps{};
        double         preprocessing_total_ms{};
        double         frontend_total_ms{};
        backend_sample local_mapping;
        backend_sample loop_closing;
        backend_sample global_bundle_adjustment;
    };

    /** Records a backend job on scope exit, including aborted jobs and early returns. */
    class backend_timer
    {
      public:
        backend_timer(processing_statistics &statistics, backend_stage stage) : _statistics(statistics), _stage(stage) {}
        backend_timer(const backend_timer &)            = delete;
        backend_timer &operator=(const backend_timer &) = delete;

        ~backend_timer()
        {
            _statistics.record_backend(_stage, std::chrono::duration<double, std::milli>(clock::now() - _started).count());
        }

      private:
        processing_statistics &_statistics;
        backend_stage          _stage;
        clock::time_point      _started{clock::now()};
    };

    /** Begin a playback session before submitting any work. */
    void start(clock::time_point started = clock::now())
    {
        const std::lock_guard lock(_mutex);
        _started  = started;
        _snapshot = {};
    }

    void record_frame(frame_sample sample, clock::time_point completed = clock::now())
    {
        const std::lock_guard lock(_mutex);
        _snapshot.frame = sample;
        ++_snapshot.frames;
        _snapshot.preprocessing_total_ms += sample.preprocessing_ms;
        _snapshot.frontend_total_ms += sample.frontend_ms;
        _snapshot.elapsed_seconds = std::chrono::duration<double>(completed - _started).count();
        _snapshot.throughput_fps = _snapshot.elapsed_seconds > 0.0 ? static_cast<double>(_snapshot.frames) / _snapshot.elapsed_seconds : 0.0;
    }

    void record_backend(backend_stage stage, double milliseconds)
    {
        const std::lock_guard lock(_mutex);
        auto &sample = stage == backend_stage::LOCAL_MAPPING ? _snapshot.local_mapping
                       : stage == backend_stage::LOOP_CLOSING ? _snapshot.loop_closing
                                                            : _snapshot.global_bundle_adjustment;
        ++sample.jobs;
        sample.last_ms = milliseconds;
        sample.total_ms += milliseconds;
    }

    /** Returns a value copy; consumers never hold a lock while logging or plotting. */
    snapshot get_snapshot() const
    {
        const std::lock_guard lock(_mutex);
        return _snapshot;
    }

  private:
    mutable std::mutex _mutex;
    clock::time_point  _started{clock::now()};
    snapshot           _snapshot;
};

} // namespace ORB_SLAM3