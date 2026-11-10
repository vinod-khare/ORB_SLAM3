#include "dbow3_vocabulary.h"

#include <fstream>
#include <stdexcept>

#include <DBow3/Vocabulary.h>

namespace ORB_SLAM3
{
class binary_dbow3_vocabulary : public DBoW3::Vocabulary
{
  public:
    std::array<int, 4> parameters() const { return {m_k, m_L, static_cast<int>(m_scoring), static_cast<int>(m_weighting)}; }

    void visit_nodes(const dbow3_vocabulary::node_visitor &visitor) const
    {
        for (const auto &node : m_nodes)
        {
            visitor(node.id, node.parent, node.weight, node.descriptor, node.children, node.word_id);
        }
    }
};

dbow3_vocabulary::dbow3_vocabulary(const std::string &filename) : _vocabulary(std::make_unique<binary_dbow3_vocabulary>())
{
    std::ifstream stream(filename, std::ios::binary);
    if (!stream || stream.peek() == std::char_traits<char>::eof())
    {
        throw std::runtime_error("Cannot open binary vocabulary: " + filename);
    }
    _vocabulary->fromStream(stream);
    const int descriptor_type = _vocabulary->getDescritorType();
    if (!stream || _vocabulary->empty() || (descriptor_type != CV_8U && descriptor_type != CV_32F) || _vocabulary->getDescritorSize() <= 0)
    {
        throw std::runtime_error("Place recognition requires a nonempty vocabulary with CV_8U or CV_32F descriptors: " + filename);
    }
}

dbow3_vocabulary::~dbow3_vocabulary() = default;

std::array<int, 4> dbow3_vocabulary::parameters() const { return _vocabulary->parameters(); }

void dbow3_vocabulary::visit_nodes(const node_visitor &visitor) const { _vocabulary->visit_nodes(visitor); }

unsigned int dbow3_vocabulary::size() const { return _vocabulary->size(); }

int dbow3_vocabulary::descriptor_type() const { return _vocabulary->getDescritorType(); }

int dbow3_vocabulary::descriptor_size() const { return _vocabulary->getDescritorSize(); }

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
