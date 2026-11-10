/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef KEY_FRAME_DATABASE_H
#define KEY_FRAME_DATABASE_H

#include "key_frame.h"
#include "frame.h"
#include "map.h"
#include "place_vocabulary.h"

#include <list>
#include <mutex>
#include <set>
#include <vector>

namespace ORB_SLAM3
{

class key_frame;
class frame;
class map;

class key_frame_database
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    key_frame_database() {}
    key_frame_database(const place_vocabulary &voc);

    void add(key_frame *pKF);

    void erase(key_frame *pKF);

    void clear();
    void clear_map(map *pMap);

    // Loop Detection(DEPRECATED)
    std::vector<key_frame *> detect_loop_candidates(key_frame *pKF, float minScore);

    // Loop and Merge Detection
    void detect_candidates(key_frame *pKF, float minScore, vector<key_frame *> &vpLoopCand, vector<key_frame *> &vpMergeCand);
    void detect_best_candidates(key_frame *pKF, vector<key_frame *> &vpLoopCand, vector<key_frame *> &vpMergeCand, int nMinWords);
    void detect_n_best_candidates(key_frame *pKF, vector<key_frame *> &vpLoopCand, vector<key_frame *> &vpMergeCand, int nNumCandidates);

    // Relocalization
    std::vector<key_frame *> detect_relocalization_candidates(frame *F, map *pMap);

  protected:
    // Associated vocabulary
    const place_vocabulary *vocabulary;

    // Inverted file
    std::vector<list<key_frame *>> inverted_file;

    // Mutex
    std::mutex database_mutex;
};

} // namespace ORB_SLAM3

#endif
