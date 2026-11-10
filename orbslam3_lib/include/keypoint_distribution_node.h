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
    keypoint_distribution_node() : is_terminal(false) {}

    void divide_node(keypoint_distribution_node &n1, keypoint_distribution_node &n2, keypoint_distribution_node &n3, keypoint_distribution_node &n4);

    std::vector<keypoint>                           keypoints;
    cv::Point2i                                     upper_left;
    cv::Point2i                                     upper_right;
    cv::Point2i                                     bottom_left;
    cv::Point2i                                     bottom_right;
    std::list<keypoint_distribution_node>::iterator node_iterator;
    bool                                            is_terminal;
};

} // namespace ORB_SLAM3