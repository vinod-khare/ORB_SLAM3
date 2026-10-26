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

#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>
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
        std::ifstream stream(filename);
        std::string   line;
        std::getline(stream, line);
        std::istringstream header(line);
        int                scoring{};
        int                weighting{};
        if (!(header >> m_k >> m_L >> scoring >> weighting) || m_k < 2 || m_k > 20 || m_L < 1 || m_L > 10 || scoring < 0 || scoring > 5 || weighting < 0 || weighting > 3)
        {
            throw std::runtime_error("Invalid ORB text vocabulary header: " + filename);
        }
        m_scoring   = static_cast<DBoW2::ScoringType>(scoring);
        m_weighting = static_cast<DBoW2::WeightingType>(weighting);
        createScoringObject();
        m_nodes.clear();
        m_words.clear();
        m_nodes.reserve(static_cast<std::size_t>((std::pow(m_k, m_L + 1) - 1) / (m_k - 1)));
        m_nodes.resize(1);
        m_nodes[0].id = 0;
        std::vector<unsigned int> word_nodes;
        while (std::getline(stream, line))
        {
            if (line.find_first_not_of(" \t\r") == std::string::npos)
            {
                continue;
            }
            std::istringstream node_stream(line);
            unsigned int       parent{};
            int                leaf{};
            const auto         id = static_cast<unsigned int>(m_nodes.size());
            if (!(node_stream >> parent >> leaf) || parent >= id || (leaf != 0 && leaf != 1))
            {
                throw std::runtime_error("Invalid ORB vocabulary node: " + filename);
            }
            m_nodes.emplace_back();
            auto &node  = m_nodes.back();
            node.id     = id;
            node.parent = parent;
            m_nodes[parent].children.push_back(id);
            node.descriptor.create(1, DBoW2::FORB::L, CV_8U);
            for (int column = 0; column < DBoW2::FORB::L; ++column)
            {
                int value{};
                if (!(node_stream >> value) || value < 0 || value > 255)
                {
                    throw std::runtime_error("Invalid ORB vocabulary descriptor: " + filename);
                }
                node.descriptor.at<unsigned char>(0, column) = static_cast<unsigned char>(value);
            }
            if (!(node_stream >> node.weight) || !std::isfinite(node.weight))
            {
                throw std::runtime_error("Invalid ORB vocabulary weight: " + filename);
            }
            if (leaf)
            {
                node.word_id = static_cast<unsigned int>(word_nodes.size());
                word_nodes.push_back(id);
            }
            else
            {
                node.children.reserve(m_k);
            }
        }
        for (const auto id : word_nodes)
        {
            m_words.push_back(&m_nodes[id]);
        }
        if (stream.bad() || empty())
        {
            throw std::runtime_error("Failed to read nonempty ORB vocabulary: " + filename);
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
    // Both loaders preserve text-file word/node IDs, so existing vectors remain valid.
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
