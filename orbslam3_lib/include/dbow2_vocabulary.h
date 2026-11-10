#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <DBoW2/TemplatedVocabulary.h>

namespace ORB_SLAM3
{
class cv_mat_descriptor
{
  public:
    using TDescriptor = cv::Mat;
    using pDescriptor = const TDescriptor *;

    static void        meanValue(const std::vector<pDescriptor> &descriptors, TDescriptor &mean);
    static double      distance(const TDescriptor &first, const TDescriptor &second);
    static std::string toString(const TDescriptor &descriptor);
    static void        fromString(TDescriptor &descriptor, const std::string &text);
};

class dbow2_vocabulary : public DBoW2::TemplatedVocabulary<cv::Mat, cv_mat_descriptor>
{
  public:
    explicit dbow2_vocabulary(const std::string &filename);

    int descriptor_type() const { return _descriptor_type; }
    int descriptor_size() const { return _descriptor_size; }

  private:
    int _descriptor_type{};
    int _descriptor_size{};
};
} // namespace ORB_SLAM3