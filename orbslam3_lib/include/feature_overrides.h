#pragma once

#include <optional>
#include <string>

namespace ORB_SLAM3
{

struct feature_overrides
{
    std::string         detector;
    std::string         descriptor;
    std::string         matcher;
    std::optional<int>  dbow_version;
    std::optional<bool> use_optical_flow;
};

} // namespace ORB_SLAM3