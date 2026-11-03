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

#include "dbow2_vocabulary.h"
#include "dbow3_vocabulary.h"

#include <mutex>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

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
