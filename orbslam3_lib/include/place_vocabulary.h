#pragma once

#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>

// Legacy core headers rely on unqualified standard-library names.
using namespace std;

namespace ORB_SLAM3
{
class dbow2_vocabulary;
class dbow3_vocabulary;

/** Runtime-selectable place recognition with stable DBoW2 vector/atlas types. */
class place_vocabulary
{
  public:
    explicit place_vocabulary(const std::string &filename, int dbow_version = 2);
    ~place_vocabulary();

    unsigned int size() const;
    void         transform(const std::vector<cv::Mat> &features, DBoW2::BowVector &bow, DBoW2::FeatureVector &nodes, int levels_up) const;
    double       score(const DBoW2::BowVector &left, const DBoW2::BowVector &right) const;
    int          version() const;
    void         set_version(int dbow_version);

  private:
    std::string                       _filename;
    int                               _version{0};
    mutable std::shared_mutex         _mutex;
    std::unique_ptr<dbow2_vocabulary> _dbow2;
    std::unique_ptr<dbow3_vocabulary> _dbow3;
};
} // namespace ORB_SLAM3
