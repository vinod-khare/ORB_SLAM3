#pragma once

#include <string>

namespace ORB_SLAM3
{

struct feature_overrides
{
    std::string detector;
    std::string descriptor;
    std::string matcher;
};

} // namespace ORB_SLAM3