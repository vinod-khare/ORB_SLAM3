#include "dbow3_vocabulary.h"

#include <stdexcept>

#include <DBow3/Vocabulary.h>

namespace ORB_SLAM3
{
dbow3_vocabulary::dbow3_vocabulary(const std::string &filename) : _vocabulary(std::make_unique<DBoW3::Vocabulary>(filename))
{
    if (_vocabulary->empty() || _vocabulary->getDescritorType() != CV_8U || _vocabulary->getDescritorSize() != 32)
    {
        throw std::runtime_error("DBoW3 requires a nonempty ORB vocabulary: " + filename);
    }
}

dbow3_vocabulary::~dbow3_vocabulary() = default;

unsigned int dbow3_vocabulary::size() const { return _vocabulary->size(); }

void dbow3_vocabulary::transform(const std::vector<cv::Mat> &features, std::map<unsigned int, double> &bow, std::map<unsigned int, std::vector<unsigned int>> &nodes, int levels_up) const
{
    DBoW3::BowVector     result;
    DBoW3::FeatureVector feature_nodes;
    _vocabulary->transform(features, result, feature_nodes, levels_up);
    bow   = std::move(result);
    nodes = std::move(feature_nodes);
}

double dbow3_vocabulary::score(const std::map<unsigned int, double> &left, const std::map<unsigned int, double> &right) const
{
    DBoW3::BowVector left_bow;
    DBoW3::BowVector right_bow;
    left_bow.insert(left.begin(), left.end());
    right_bow.insert(right.begin(), right.end());
    return _vocabulary->score(left_bow, right_bow);
}
} // namespace ORB_SLAM3
