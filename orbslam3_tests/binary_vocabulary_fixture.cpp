#include <string>

#if ORB_SLAM3_HAS_DBOW3
#include <DBow3/Vocabulary.h>
#else
#include <cstdint>
#include <fstream>

#include <opencv2/core.hpp>
#endif

void write_vocabulary_fixture(const std::string &filename, int scoring, int weighting, int descriptor_type, int descriptor_size, int levels)
{
#if ORB_SLAM3_HAS_DBOW3
    class fixture_vocabulary : public DBoW3::Vocabulary
    {
      public:
        fixture_vocabulary(int scoring, int weighting, int descriptor_type, int descriptor_size, int levels)
        {
            m_k         = 2;
            m_L         = levels;
            m_scoring   = static_cast<DBoW3::ScoringType>(scoring);
            m_weighting = static_cast<DBoW3::WeightingType>(weighting);

            createScoringObject();

            const auto first_leaf = (1u << levels) - 1;

            m_nodes.resize((1u << (levels + 1)) - 1);

            for (unsigned int id = 1; id < m_nodes.size(); ++id)
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

                if (id >= first_leaf)
                {
                    node.word_id = id - first_leaf;
                    m_words.push_back(&node);
                }
            }
        }
    };

    fixture_vocabulary(scoring, weighting, descriptor_type, descriptor_size, levels).save(filename, true);
#else
    std::ofstream stream(filename, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    const auto write = [&]<typename VALUE_T_>(const VALUE_T_ &value) { stream.write(reinterpret_cast<const char *>(&value), sizeof(value)); };
    write(std::uint64_t{88877711233});
    write(std::uint8_t{0});
    const auto nodes      = (1u << (levels + 1)) - 1;
    const auto first_leaf = (1u << levels) - 1;
    write(nodes);
    write(2);
    write(levels);
    write(scoring);
    write(weighting);
    for (unsigned int id = 1; id < nodes; ++id)
    {
        write(id);
        write((id - 1) / 2);
        write(0.25 * (id % 4 + 1));
        write(descriptor_size);
        write(1);
        write(descriptor_type);
        for (int column = 0; column < descriptor_size; ++column)
        {
            const auto value = static_cast<unsigned char>((id * 37 + column * 13) % 256);
            if (descriptor_type == CV_8U)
            {
                write(value);
            }
            else
            {
                write(static_cast<float>(value));
            }
        }
    }
    write(nodes - first_leaf);
    for (unsigned int id = first_leaf; id < nodes; ++id)
    {
        write(id - first_leaf);
        write(id);
    }
#endif
}