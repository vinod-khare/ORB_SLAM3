#pragma once

#include <string>

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

namespace ORB_SLAM3
{
class dbow2_vocabulary : public DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
{
  public:
    explicit dbow2_vocabulary(const std::string &filename);
};
} // namespace ORB_SLAM3