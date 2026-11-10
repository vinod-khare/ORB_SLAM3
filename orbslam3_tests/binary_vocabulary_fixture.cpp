#include <string>

#include <DBow3/Vocabulary.h>

void write_vocabulary_fixture(const std::string &filename, int scoring, int weighting, int descriptor_type, int descriptor_size)
{
    class fixture_vocabulary : public DBoW3::Vocabulary
    {
      public:
        fixture_vocabulary(int scoring, int weighting, int descriptor_type, int descriptor_size)
        {
            m_k         = 2;
            m_L         = 6;
            m_scoring   = static_cast<DBoW3::ScoringType>(scoring);
            m_weighting = static_cast<DBoW3::WeightingType>(weighting);
            createScoringObject();
            m_nodes.resize(127);
            for (unsigned int id = 1; id < 127; ++id)
            {
                auto &node  = m_nodes[id];
                node.id     = id;
                node.parent = (id - 1) / 2;
                m_nodes[node.parent].children.push_back(id);
                node.descriptor.create(1, descriptor_size, CV_8U);
                for (int column = 0; column < descriptor_size; ++column)
                {
                    node.descriptor.at<unsigned char>(0, column) = static_cast<unsigned char>((id * 37 + column * 13) % 256);
                }
                node.descriptor.convertTo(node.descriptor, descriptor_type);
                node.weight = 0.25 * (id % 4 + 1);
                if (id >= 63)
                {
                    node.word_id = id - 63;
                    m_words.push_back(&node);
                }
            }
        }
    };
    fixture_vocabulary(scoring, weighting, descriptor_type, descriptor_size).save(filename, true);
}