#pragma once

#include "keypoint.h"

#include <list>
#include <vector>

#include <opencv2/features2d.hpp>

namespace ORB_SLAM3
{

class keypoint_distribution_node
{
  public:
    keypoint_distribution_node() : bNoMore(false) {}

    void divide_node(keypoint_distribution_node &n1, keypoint_distribution_node &n2, keypoint_distribution_node &n3, keypoint_distribution_node &n4);

    std::vector<keypoint>                           vKeys;
    cv::Point2i                                     UL, UR, BL, BR;
    std::list<keypoint_distribution_node>::iterator lit;
    bool                                            bNoMore;
};

} // namespace ORB_SLAM3