#pragma once

namespace ORB_SLAM3
{
class viewer_control
{
  public:
    virtual ~viewer_control() = default;

    virtual void set_step_by_step(bool enabled) = 0;
    virtual void process_next_frame() = 0;
    virtual void set_localization_mode(bool enabled) = 0;
    virtual void reset_active_map() = 0;
};
} // namespace ORB_SLAM3
