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

#include "key_frame_database.h"

#include "key_frame.h"

#include <mutex>
#include <utility>

#include <DBoW2/BowVector.h>

using namespace std;

namespace ORB_SLAM3
{

key_frame_database::key_frame_database(const place_vocabulary &voc) : vocabulary(&voc) { inverted_file.resize(voc.size()); }

void key_frame_database::add(key_frame *pKF)
{
    unique_lock<mutex> lock(database_mutex);

    for (DBoW2::BowVector::const_iterator vit = pKF->bow_vector.begin(), vend = pKF->bow_vector.end(); vit != vend; vit++)
    {
        inverted_file[vit->first].push_back(pKF);
    }
}

void key_frame_database::erase(key_frame *pKF)
{
    unique_lock<mutex> lock(database_mutex);

    // Erase elements in the Inverse File for the entry
    for (DBoW2::BowVector::const_iterator vit = pKF->bow_vector.begin(), vend = pKF->bow_vector.end(); vit != vend; vit++)
    {
        // List of keyframes that share the word
        list<key_frame *> &lKFs = inverted_file[vit->first];

        for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
        {
            if (pKF == *lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void key_frame_database::clear()
{
    inverted_file.clear();
    inverted_file.resize(vocabulary->size());
}

void key_frame_database::clear_map(map *pMap)
{
    unique_lock<mutex> lock(database_mutex);

    // Erase elements in the Inverse File for the entry
    for (std::vector<list<key_frame *>>::iterator vit = inverted_file.begin(), vend = inverted_file.end(); vit != vend; vit++)
    {
        // List of keyframes that share the word
        list<key_frame *> &lKFs = *vit;

        for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend;)
        {
            key_frame *pKFi = *lit;
            if (pMap == pKFi->get_map())
            {
                lit = lKFs.erase(lit);
                // Dont delete the KF because the class Map clean all the KF when it is destroyed
            }
            else
            {
                ++lit;
            }
        }
    }
}

vector<key_frame *> key_frame_database::detect_loop_candidates(key_frame *pKF, float minScore)
{
    set<key_frame *>  spConnectedKeyFrames = pKF->get_connected_key_frames();
    list<key_frame *> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(database_mutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->bow_vector.begin(), vend = pKF->bow_vector.end(); vit != vend; vit++)
        {
            list<key_frame *> &lKFs = inverted_file[vit->first];

            for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                key_frame *pKFi = *lit;
                if (pKFi->get_map() == pKF->get_map()) // For consider a loop candidate it a candidate it must be in the same map
                {
                    if (pKFi->loop_query_id != pKF->id)
                    {
                        pKFi->loop_common_words = 0;
                        if (!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->loop_query_id = pKF->id;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }
                    pKFi->loop_common_words++;
                }
            }
        }
    }

    if (lKFsSharingWords.empty())
    {
        return vector<key_frame *>();
    }

    list<pair<float, key_frame *>> lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->loop_common_words > maxCommonWords)
        {
            maxCommonWords = (*lit)->loop_common_words;
        }
    }

    int minCommonWords = maxCommonWords * 0.8f;

    int nscores = 0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        key_frame *pKFi = *lit;

        if (pKFi->loop_common_words > minCommonWords)
        {
            nscores++;

            float si = vocabulary->score(pKF->bow_vector, pKFi->bow_vector);

            pKFi->loop_score = si;
            if (si >= minScore)
            {
                lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }
    }

    if (lScoreAndMatch.empty())
    {
        return vector<key_frame *>();
    }

    list<pair<float, key_frame *>> lAccScoreAndMatch;
    float                         bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for (list<pair<float, key_frame *>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        key_frame          *pKFi     = it->second;
        vector<key_frame *> vpNeighs = pKFi->get_best_covisibility_key_frames(10);

        float     bestScore = it->first;
        float     accScore  = it->first;
        key_frame *pBestKF   = pKFi;
        for (vector<key_frame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            key_frame *pKF2 = *vit;
            if (pKF2->loop_query_id == pKF->id && pKF2->loop_common_words > minCommonWords)
            {
                accScore += pKF2->loop_score;
                if (pKF2->loop_score > bestScore)
                {
                    pBestKF   = pKF2;
                    bestScore = pKF2->loop_score;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
        {
            bestAccScore = accScore;
        }
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f * bestAccScore;

    set<key_frame *>    spAlreadyAddedKF;
    vector<key_frame *> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (list<pair<float, key_frame *>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        if (it->first > minScoreToRetain)
        {
            key_frame *pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}

void key_frame_database::detect_candidates(key_frame *pKF, float minScore, vector<key_frame *> &vpLoopCand, vector<key_frame *> &vpMergeCand)
{
    set<key_frame *>  spConnectedKeyFrames = pKF->get_connected_key_frames();
    list<key_frame *> lKFsSharingWordsLoop, lKFsSharingWordsMerge;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(database_mutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->bow_vector.begin(), vend = pKF->bow_vector.end(); vit != vend; vit++)
        {
            list<key_frame *> &lKFs = inverted_file[vit->first];

            for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                key_frame *pKFi = *lit;
                if (pKFi->get_map() == pKF->get_map()) // For consider a loop candidate it a candidate it must be in the same map
                {
                    if (pKFi->loop_query_id != pKF->id)
                    {
                        pKFi->loop_common_words = 0;
                        if (!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->loop_query_id = pKF->id;
                            lKFsSharingWordsLoop.push_back(pKFi);
                        }
                    }
                    pKFi->loop_common_words++;
                }
                else if (!pKFi->get_map()->is_bad())
                {
                    if (pKFi->merge_query_id != pKF->id)
                    {
                        pKFi->merge_words = 0;
                        if (!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->merge_query_id = pKF->id;
                            lKFsSharingWordsMerge.push_back(pKFi);
                        }
                    }
                    pKFi->merge_words++;
                }
            }
        }
    }

    if (lKFsSharingWordsLoop.empty() && lKFsSharingWordsMerge.empty())
    {
        return;
    }

    if (!lKFsSharingWordsLoop.empty())
    {
        list<pair<float, key_frame *>> lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;
        for (list<key_frame *>::iterator lit = lKFsSharingWordsLoop.begin(), lend = lKFsSharingWordsLoop.end(); lit != lend; lit++)
        {
            if ((*lit)->loop_common_words > maxCommonWords)
            {
                maxCommonWords = (*lit)->loop_common_words;
            }
        }

        int minCommonWords = maxCommonWords * 0.8f;

        int nscores = 0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for (list<key_frame *>::iterator lit = lKFsSharingWordsLoop.begin(), lend = lKFsSharingWordsLoop.end(); lit != lend; lit++)
        {
            key_frame *pKFi = *lit;

            if (pKFi->loop_common_words > minCommonWords)
            {
                nscores++;

                float si = vocabulary->score(pKF->bow_vector, pKFi->bow_vector);

                pKFi->loop_score = si;
                if (si >= minScore)
                {
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
                }
            }
        }

        if (!lScoreAndMatch.empty())
        {
            list<pair<float, key_frame *>> lAccScoreAndMatch;
            float                         bestAccScore = minScore;

            // Lets now accumulate score by covisibility
            for (list<pair<float, key_frame *>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
            {
                key_frame          *pKFi     = it->second;
                vector<key_frame *> vpNeighs = pKFi->get_best_covisibility_key_frames(10);

                float     bestScore = it->first;
                float     accScore  = it->first;
                key_frame *pBestKF   = pKFi;
                for (vector<key_frame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
                {
                    key_frame *pKF2 = *vit;
                    if (pKF2->loop_query_id == pKF->id && pKF2->loop_common_words > minCommonWords)
                    {
                        accScore += pKF2->loop_score;
                        if (pKF2->loop_score > bestScore)
                        {
                            pBestKF   = pKF2;
                            bestScore = pKF2->loop_score;
                        }
                    }
                }

                lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
                if (accScore > bestAccScore)
                {
                    bestAccScore = accScore;
                }
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f * bestAccScore;

            set<key_frame *> spAlreadyAddedKF;
            vpLoopCand.reserve(lAccScoreAndMatch.size());

            for (list<pair<float, key_frame *>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
            {
                if (it->first > minScoreToRetain)
                {
                    key_frame *pKFi = it->second;
                    if (!spAlreadyAddedKF.count(pKFi))
                    {
                        vpLoopCand.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
        }
    }

    if (!lKFsSharingWordsMerge.empty())
    {
        list<pair<float, key_frame *>> lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;
        for (list<key_frame *>::iterator lit = lKFsSharingWordsMerge.begin(), lend = lKFsSharingWordsMerge.end(); lit != lend; lit++)
        {
            if ((*lit)->merge_words > maxCommonWords)
            {
                maxCommonWords = (*lit)->merge_words;
            }
        }

        int minCommonWords = maxCommonWords * 0.8f;

        int nscores = 0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for (list<key_frame *>::iterator lit = lKFsSharingWordsMerge.begin(), lend = lKFsSharingWordsMerge.end(); lit != lend; lit++)
        {
            key_frame *pKFi = *lit;

            if (pKFi->merge_words > minCommonWords)
            {
                nscores++;

                float si = vocabulary->score(pKF->bow_vector, pKFi->bow_vector);

                pKFi->merge_score = si;
                if (si >= minScore)
                {
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
                }
            }
        }

        if (!lScoreAndMatch.empty())
        {
            list<pair<float, key_frame *>> lAccScoreAndMatch;
            float                         bestAccScore = minScore;

            // Lets now accumulate score by covisibility
            for (list<pair<float, key_frame *>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
            {
                key_frame          *pKFi     = it->second;
                vector<key_frame *> vpNeighs = pKFi->get_best_covisibility_key_frames(10);

                float     bestScore = it->first;
                float     accScore  = it->first;
                key_frame *pBestKF   = pKFi;
                for (vector<key_frame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
                {
                    key_frame *pKF2 = *vit;
                    if (pKF2->merge_query_id == pKF->id && pKF2->merge_words > minCommonWords)
                    {
                        accScore += pKF2->merge_score;
                        if (pKF2->merge_score > bestScore)
                        {
                            pBestKF   = pKF2;
                            bestScore = pKF2->merge_score;
                        }
                    }
                }

                lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
                if (accScore > bestAccScore)
                {
                    bestAccScore = accScore;
                }
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f * bestAccScore;

            set<key_frame *> spAlreadyAddedKF;
            vpMergeCand.reserve(lAccScoreAndMatch.size());

            for (list<pair<float, key_frame *>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
            {
                if (it->first > minScoreToRetain)
                {
                    key_frame *pKFi = it->second;
                    if (!spAlreadyAddedKF.count(pKFi))
                    {
                        vpMergeCand.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
        }
    }

    for (DBoW2::BowVector::const_iterator vit = pKF->bow_vector.begin(), vend = pKF->bow_vector.end(); vit != vend; vit++)
    {
        list<key_frame *> &lKFs = inverted_file[vit->first];

        for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
        {
            key_frame *pKFi     = *lit;
            pKFi->loop_query_id  = -1;
            pKFi->merge_query_id = -1;
        }
    }
}

void key_frame_database::detect_best_candidates(key_frame *pKF, vector<key_frame *> &vpLoopCand, vector<key_frame *> &vpMergeCand, int nMinWords)
{
    list<key_frame *> lKFsSharingWords;
    set<key_frame *>  spConnectedKF;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(database_mutex);

        spConnectedKF = pKF->get_connected_key_frames();

        for (DBoW2::BowVector::const_iterator vit = pKF->bow_vector.begin(), vend = pKF->bow_vector.end(); vit != vend; vit++)
        {
            list<key_frame *> &lKFs = inverted_file[vit->first];

            for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                key_frame *pKFi = *lit;
                if (spConnectedKF.find(pKFi) != spConnectedKF.end())
                {
                    continue;
                }
                if (pKFi->place_recognition_query_id != pKF->id)
                {
                    pKFi->place_recognition_words = 0;
                    pKFi->place_recognition_query_id = pKF->id;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->place_recognition_words++;
            }
        }
    }
    if (lKFsSharingWords.empty())
    {
        return;
    }

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->place_recognition_words > maxCommonWords)
        {
            maxCommonWords = (*lit)->place_recognition_words;
        }
    }

    int minCommonWords = maxCommonWords * 0.8f;

    if (minCommonWords < nMinWords)
    {
        minCommonWords = nMinWords;
    }

    list<pair<float, key_frame *>> lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        key_frame *pKFi = *lit;

        if (pKFi->place_recognition_words > minCommonWords)
        {
            nscores++;
            float si                     = vocabulary->score(pKF->bow_vector, pKFi->bow_vector);
            pKFi->place_recognition_score = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
    {
        return;
    }

    list<pair<float, key_frame *>> lAccScoreAndMatch;
    float                         bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, key_frame *>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        key_frame          *pKFi     = it->second;
        vector<key_frame *> vpNeighs = pKFi->get_best_covisibility_key_frames(10);

        float     bestScore = it->first;
        float     accScore  = bestScore;
        key_frame *pBestKF   = pKFi;
        for (vector<key_frame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            key_frame *pKF2 = *vit;
            if (pKF2->place_recognition_query_id != pKF->id)
            {
                continue;
            }

            accScore += pKF2->place_recognition_score;
            if (pKF2->place_recognition_score > bestScore)
            {
                pBestKF   = pKF2;
                bestScore = pKF2->place_recognition_score;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
        {
            bestAccScore = accScore;
        }
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float           minScoreToRetain = 0.75f * bestAccScore;
    set<key_frame *> spAlreadyAddedKF;
    vpLoopCand.reserve(lAccScoreAndMatch.size());
    vpMergeCand.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, key_frame *>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        const float &si = it->first;
        if (si > minScoreToRetain)
        {
            key_frame *pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                if (pKF->get_map() == pKFi->get_map())
                {
                    vpLoopCand.push_back(pKFi);
                }
                else
                {
                    vpMergeCand.push_back(pKFi);
                }
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }
}

bool compFirst(const pair<float, key_frame *> &a, const pair<float, key_frame *> &b) { return a.first > b.first; }

void key_frame_database::detect_n_best_candidates(key_frame *pKF, vector<key_frame *> &vpLoopCand, vector<key_frame *> &vpMergeCand, int nNumCandidates)
{
    list<key_frame *> lKFsSharingWords;
    set<key_frame *>  spConnectedKF;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(database_mutex);

        spConnectedKF = pKF->get_connected_key_frames();

        for (DBoW2::BowVector::const_iterator vit = pKF->bow_vector.begin(), vend = pKF->bow_vector.end(); vit != vend; vit++)
        {
            list<key_frame *> &lKFs = inverted_file[vit->first];

            for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                key_frame *pKFi = *lit;

                if (pKFi->place_recognition_query_id != pKF->id)
                {
                    pKFi->place_recognition_words = 0;
                    if (!spConnectedKF.count(pKFi))
                    {

                        pKFi->place_recognition_query_id = pKF->id;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->place_recognition_words++;
            }
        }
    }
    if (lKFsSharingWords.empty())
    {
        return;
    }

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->place_recognition_words > maxCommonWords)
        {
            maxCommonWords = (*lit)->place_recognition_words;
        }
    }

    int minCommonWords = maxCommonWords * 0.8f;

    list<pair<float, key_frame *>> lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        key_frame *pKFi = *lit;

        if (pKFi->place_recognition_words > minCommonWords)
        {
            nscores++;
            float si                     = vocabulary->score(pKF->bow_vector, pKFi->bow_vector);
            pKFi->place_recognition_score = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
    {
        return;
    }

    list<pair<float, key_frame *>> lAccScoreAndMatch;
    float                         bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, key_frame *>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        key_frame          *pKFi     = it->second;
        vector<key_frame *> vpNeighs = pKFi->get_best_covisibility_key_frames(10);

        float     bestScore = it->first;
        float     accScore  = bestScore;
        key_frame *pBestKF   = pKFi;
        for (vector<key_frame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            key_frame *pKF2 = *vit;
            if (pKF2->place_recognition_query_id != pKF->id)
            {
                continue;
            }

            accScore += pKF2->place_recognition_score;
            if (pKF2->place_recognition_score > bestScore)
            {
                pBestKF   = pKF2;
                bestScore = pKF2->place_recognition_score;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
        {
            bestAccScore = accScore;
        }
    }

    lAccScoreAndMatch.sort(compFirst);

    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<key_frame *>                         spAlreadyAddedKF;
    int                                     i  = 0;
    list<pair<float, key_frame *>>::iterator it = lAccScoreAndMatch.begin();
    while (std::cmp_less(i, lAccScoreAndMatch.size()) && (std::cmp_less(vpLoopCand.size(), nNumCandidates) || std::cmp_less(vpMergeCand.size(), nNumCandidates)))
    {
        key_frame *pKFi = it->second;
        if (pKFi->is_bad())
        {
            continue;
        }

        if (!spAlreadyAddedKF.count(pKFi))
        {
            if (pKF->get_map() == pKFi->get_map() && std::cmp_less(vpLoopCand.size(), nNumCandidates))
            {
                vpLoopCand.push_back(pKFi);
            }
            else if (pKF->get_map() != pKFi->get_map() && std::cmp_less(vpMergeCand.size(), nNumCandidates) && !pKFi->get_map()->is_bad())
            {
                vpMergeCand.push_back(pKFi);
            }
            spAlreadyAddedKF.insert(pKFi);
        }
        i++;
        it++;
    }
}

vector<key_frame *> key_frame_database::detect_relocalization_candidates(frame *F, map *pMap)
{
    list<key_frame *> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(database_mutex);

        for (DBoW2::BowVector::const_iterator vit = F->bow_vector.begin(), vend = F->bow_vector.end(); vit != vend; vit++)
        {
            list<key_frame *> &lKFs = inverted_file[vit->first];

            for (list<key_frame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                key_frame *pKFi = *lit;
                if (pKFi->relocalization_query_id != F->id)
                {
                    pKFi->relocalization_words = 0;
                    pKFi->relocalization_query_id = F->id;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->relocalization_words++;
            }
        }
    }
    if (lKFsSharingWords.empty())
    {
        return vector<key_frame *>();
    }

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->relocalization_words > maxCommonWords)
        {
            maxCommonWords = (*lit)->relocalization_words;
        }
    }

    int minCommonWords = maxCommonWords * 0.8f;

    list<pair<float, key_frame *>> lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    for (list<key_frame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        key_frame *pKFi = *lit;

        if (pKFi->relocalization_words > minCommonWords)
        {
            nscores++;
            float si          = vocabulary->score(F->bow_vector, pKFi->bow_vector);
            pKFi->relocalization_score = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
    {
        return vector<key_frame *>();
    }

    list<pair<float, key_frame *>> lAccScoreAndMatch;
    float                         bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, key_frame *>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        key_frame          *pKFi     = it->second;
        vector<key_frame *> vpNeighs = pKFi->get_best_covisibility_key_frames(10);

        float     bestScore = it->first;
        float     accScore  = bestScore;
        key_frame *pBestKF   = pKFi;
        for (vector<key_frame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            key_frame *pKF2 = *vit;
            if (pKF2->relocalization_query_id != F->id)
            {
                continue;
            }

            accScore += pKF2->relocalization_score;
            if (pKF2->relocalization_score > bestScore)
            {
                pBestKF   = pKF2;
                bestScore = pKF2->relocalization_score;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
        {
            bestAccScore = accScore;
        }
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float              minScoreToRetain = 0.75f * bestAccScore;
    set<key_frame *>    spAlreadyAddedKF;
    vector<key_frame *> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, key_frame *>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        const float &si = it->first;
        if (si > minScoreToRetain)
        {
            key_frame *pKFi = it->second;
            if (pKFi->get_map() != pMap)
            {
                continue;
            }
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} // namespace ORB_SLAM3
