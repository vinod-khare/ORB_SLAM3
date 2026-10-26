#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace DBoW3
{
class Vocabulary;
}

namespace ORB_SLAM3
{
/** Isolates DBoW3 headers, whose include guards collide with DBoW2 headers. */
class dbow3_vocabulary
{
  public:
    explicit dbow3_vocabulary(const std::string &filename);
    ~dbow3_vocabulary();
    unsigned int size() const;
    void         transform(const std::vector<cv::Mat> &features, std::map<unsigned int, double> &bow, std::map<unsigned int, std::vector<unsigned int>> &nodes, int levels_up) const;
    double       score(const std::map<unsigned int, double> &left, const std::map<unsigned int, double> &right) const;

  private:
    std::unique_ptr<DBoW3::Vocabulary> _vocabulary;
};
} // namespace ORB_SLAM3
