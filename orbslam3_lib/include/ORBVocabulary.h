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


#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include<DBoW2/FORB.h>
#include<DBoW2/TemplatedVocabulary.h>

#include <fstream>
#include <sstream>
#include <cmath>

// The vendored DBoW2 pulled in "using namespace std;" transitively; the
// installed package doesn't, so restore it here since much of the codebase
// relies on unqualified std:: names being ambient.
using namespace std;

namespace ORB_SLAM3
{

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabularyBase;

// The installed DBoW2 package only supports loading vocabularies via
// cv::FileStorage (YAML/XML/gz); ORBvoc.txt uses ORB-SLAM3's own plain-text
// format, so reimplement that loader here against the base class's
// protected tree-building members instead of depending on a vendored fork.
class ORBVocabulary : public ORBVocabularyBase
{
public:
    bool loadFromTextFile(const std::string &filename)
    {
        ifstream f(filename.c_str());
        if(!f.is_open() || f.eof())
            return false;

        m_words.clear();
        m_nodes.clear();

        string s;
        getline(f, s);
        stringstream ss;
        ss << s;
        ss >> m_k;
        ss >> m_L;
        int n1, n2;
        ss >> n1;
        ss >> n2;

        if(m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3)
        {
            cerr << "Vocabulary loading failure: This is not a correct text file!" << endl;
            return false;
        }

        m_scoring = (DBoW2::ScoringType)n1;
        m_weighting = (DBoW2::WeightingType)n2;
        createScoringObject();

        int expected_nodes = (int)((pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));
        m_nodes.reserve(expected_nodes);
        m_words.reserve((size_t)pow((double)m_k, (double)m_L + 1));

        m_nodes.resize(1);
        m_nodes[0].id = 0;
        while(!f.eof())
        {
            string snode;
            getline(f, snode);
            stringstream ssnode;
            ssnode << snode;

            int nid = m_nodes.size();
            m_nodes.resize(m_nodes.size() + 1);
            m_nodes[nid].id = nid;

            int pid;
            ssnode >> pid;
            m_nodes[nid].parent = pid;
            m_nodes[pid].children.push_back(nid);

            int nIsLeaf;
            ssnode >> nIsLeaf;

            stringstream ssd;
            for(int iD = 0; iD < DBoW2::FORB::L; iD++)
            {
                string sElement;
                ssnode >> sElement;
                ssd << sElement << " ";
            }
            DBoW2::FORB::fromString(m_nodes[nid].descriptor, ssd.str());

            ssnode >> m_nodes[nid].weight;

            if(nIsLeaf > 0)
            {
                int wid = m_words.size();
                m_words.resize(wid + 1);
                m_nodes[nid].word_id = wid;
                m_words[wid] = &m_nodes[nid];
            }
            else
            {
                m_nodes[nid].children.reserve(m_k);
            }
        }

        return true;
    }
};

} //namespace ORB_SLAM

#endif // ORBVOCABULARY_H
