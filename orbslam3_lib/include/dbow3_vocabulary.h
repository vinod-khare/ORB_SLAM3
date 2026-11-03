#pragma once

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace ORB_SLAM3
{
class binary_dbow3_vocabulary;

/** Isolates DBoW3 headers, whose include guards collide with DBoW2 headers. */
class dbow3_vocabulary
{
  public:
    using node_visitor = std::function<void(unsigned int, unsigned int, double, const cv::Mat &, const std::vector<unsigned int> &, unsigned int)>;

    explicit dbow3_vocabulary(const std::string &filename);
    ~dbow3_vocabulary();
    unsigned int       size() const;
    void               transform(const std::vector<cv::Mat> &features, std::map<unsigned int, double> &bow, std::map<unsigned int, std::vector<unsigned int>> &nodes, int levels_up) const;
    double             score(const std::map<unsigned int, double> &left, const std::map<unsigned int, double> &right) const;
    std::array<int, 4> parameters() const;
    void               visit_nodes(const node_visitor &visitor) const;

  private:
    std::unique_ptr<binary_dbow3_vocabulary> _vocabulary;
};
} // namespace ORB_SLAM3