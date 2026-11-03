#include "dbow2_vocabulary.h"

#include "dbow3_vocabulary.h"

#include <stdexcept>

namespace ORB_SLAM3
{
dbow2_vocabulary::dbow2_vocabulary(const std::string &filename)
{
    const dbow3_vocabulary binary(filename);
    const auto             parameters = binary.parameters();
    m_k                               = parameters[0];
    m_L                               = parameters[1];
    m_scoring                         = static_cast<DBoW2::ScoringType>(parameters[2]);
    m_weighting                       = static_cast<DBoW2::WeightingType>(parameters[3]);
    createScoringObject();
    m_nodes.clear();
    m_words.clear();
    m_words.resize(binary.size());
    binary.visit_nodes(
        [&](unsigned int id, unsigned int parent, double weight, const cv::Mat &descriptor, const std::vector<unsigned int> &children, unsigned int word_id)
        {
            m_nodes.emplace_back(id);
            auto &node      = m_nodes.back();
            node.parent     = parent;
            node.weight     = weight;
            node.descriptor = descriptor;
            node.children   = children;
            node.word_id    = word_id;
        });
    for (auto &node : m_nodes)
    {
        if (node.isLeaf())
        {
            if (node.word_id >= m_words.size())
            {
                throw std::runtime_error("Invalid binary vocabulary word ID: " + filename);
            }
            m_words[node.word_id] = &node;
        }
    }
}
} // namespace ORB_SLAM3