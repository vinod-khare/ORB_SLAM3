/**
 * This file is part of ORB-SLAM3.
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez,
 * José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 * ORB-SLAM3 is distributed WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See <http://www.gnu.org/licenses/> for the GNU General Public License.
 */

#include "place_vocabulary.h"

#include "dbow3_vocabulary.h"

#include <mutex>
#include <stdexcept>

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{
class dbow2_vocabulary : public DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
{
  public:
    explicit dbow2_vocabulary(const std::string &filename)
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
};

place_vocabulary::place_vocabulary(const std::string &filename, int dbow_version) : _filename(filename) { set_version(dbow_version); }

place_vocabulary::~place_vocabulary() = default;

void place_vocabulary::set_version(int dbow_version)
{
    if (dbow_version != 2 && dbow_version != 3)
    {
        throw std::invalid_argument("dbow_version must be 2 or 3");
    }
    const std::unique_lock lock(_mutex);
    if (_version == dbow_version)
    {
        return;
    }
    if (dbow_version == 2)
    {
        auto vocabulary = std::make_unique<dbow2_vocabulary>(_filename);
        _dbow2          = std::move(vocabulary);
        _dbow3.reset();
    }
    else
    {
        auto vocabulary = std::make_unique<dbow3_vocabulary>(_filename);
        _dbow3          = std::move(vocabulary);
        _dbow2.reset();
    }
    _version = dbow_version;
    spdlog::info("Place recognition: DBoW{}", _version);
}

int place_vocabulary::version() const
{
    const std::shared_lock lock(_mutex);
    return _version;
}

unsigned int place_vocabulary::size() const
{
    const std::shared_lock lock(_mutex);
    return _version == 2 ? _dbow2->size() : _dbow3->size();
}

void place_vocabulary::transform(const std::vector<cv::Mat> &features, DBoW2::BowVector &bow, DBoW2::FeatureVector &nodes, int levels_up) const
{
    const std::shared_lock lock(_mutex);
    if (_version == 2)
    {
        _dbow2->transform(features, bow, nodes, levels_up);
    }
    else
    {
        _dbow3->transform(features, bow, nodes, levels_up);
    }
}

double place_vocabulary::score(const DBoW2::BowVector &left, const DBoW2::BowVector &right) const
{
    const std::shared_lock lock(_mutex);
    return _version == 2 ? _dbow2->score(left, right) : _dbow3->score(left, right);
}
} // namespace ORB_SLAM3
