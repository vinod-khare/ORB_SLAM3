#include "dbow2_vocabulary.h"

#include "dbow3_vocabulary.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ORB_SLAM3
{
void cv_mat_descriptor::meanValue(const std::vector<pDescriptor> &descriptors, TDescriptor &mean)
{
    if (descriptors.empty())
    {
        mean.release();
        return;
    }

    const auto &first = *descriptors.front();
    if (first.type() == CV_8U)
    {
        mean = cv::Mat::zeros(first.size(), first.type());
        std::vector<int> bit_counts(static_cast<std::size_t>(first.cols) * 8);
        for (const auto *descriptor : descriptors)
        {
            CV_Assert(descriptor->type() == CV_8U && descriptor->size() == first.size());
            for (int column = 0; column < first.cols; ++column)
            {
                const auto byte = descriptor->at<unsigned char>(0, column);
                for (int bit = 0; bit < 8; ++bit)
                {
                    bit_counts[static_cast<std::size_t>(column) * 8 + bit] += (byte >> bit) & 1;
                }
            }
        }
        auto      *mean_bytes = mean.ptr<unsigned char>();
        const auto threshold  = static_cast<int>((descriptors.size() + 1) / 2);
        for (int column = 0; column < first.cols; ++column)
        {
            for (int bit = 0; bit < 8; ++bit)
            {
                if (bit_counts[static_cast<std::size_t>(column) * 8 + bit] >= threshold)
                {
                    mean_bytes[column] |= static_cast<unsigned char>(1U << bit);
                }
            }
        }
        return;
    }

    if (first.type() != CV_32F)
    {
        throw std::invalid_argument("DBoW2 supports CV_8U and CV_32F descriptors");
    }
    mean = cv::Mat::zeros(first.size(), first.type());
    for (const auto *descriptor : descriptors)
    {
        CV_Assert(descriptor->type() == CV_32F && descriptor->size() == first.size());
        mean += *descriptor;
    }
    mean /= static_cast<double>(descriptors.size());
}

double cv_mat_descriptor::distance(const TDescriptor &first, const TDescriptor &second)
{
    if (first.type() != second.type() || first.size() != second.size())
    {
        throw std::invalid_argument("DBoW2 descriptor types and sizes must match");
    }
    if (first.type() == CV_8U)
    {
        return cv::norm(first, second, cv::NORM_HAMMING);
    }
    if (first.type() == CV_32F)
    {
        return cv::norm(first, second, cv::NORM_L2SQR);
    }
    throw std::invalid_argument("DBoW2 supports CV_8U and CV_32F descriptors");
}

std::string cv_mat_descriptor::toString(const TDescriptor &descriptor)
{
    std::ostringstream stream;
    if (descriptor.type() == CV_8U)
    {
        for (int column = 0; column < descriptor.cols; ++column)
        {
            stream << static_cast<int>(descriptor.at<unsigned char>(0, column)) << ' ';
        }
        return stream.str();
    }
    if (descriptor.type() != CV_32F)
    {
        throw std::invalid_argument("DBoW2 supports CV_8U and CV_32F descriptors");
    }

    stream << "cvmat " << descriptor.type() << ' ' << descriptor.cols << ' ' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (int column = 0; column < descriptor.cols; ++column)
    {
        stream << descriptor.at<float>(0, column) << ' ';
    }
    return stream.str();
}

void cv_mat_descriptor::fromString(TDescriptor &descriptor, const std::string &text)
{
    std::istringstream stream(text);
    std::string        type_marker;
    stream >> type_marker;
    if (type_marker == "cvmat")
    {
        int type{};
        int columns{};
        stream >> type >> columns;
        if (!stream || type != CV_32F || columns <= 0)
        {
            throw std::invalid_argument("Invalid floating-point DBoW2 descriptor");
        }
        descriptor.create(1, columns, type);
        for (int column = 0; column < columns; ++column)
        {
            stream >> descriptor.at<float>(0, column);
        }
        if (!stream)
        {
            throw std::invalid_argument("Incomplete floating-point DBoW2 descriptor");
        }
        return;
    }

    stream.clear();
    stream.str(text);
    std::vector<unsigned char> bytes;
    int                        value{};
    while (stream >> value)
    {
        if (value < 0 || value > 255)
        {
            throw std::invalid_argument("Invalid binary DBoW2 descriptor value");
        }
        bytes.push_back(static_cast<unsigned char>(value));
    }
    if (bytes.empty() || !stream.eof())
    {
        throw std::invalid_argument("Invalid binary DBoW2 descriptor");
    }
    descriptor.create(1, static_cast<int>(bytes.size()), CV_8U);
    std::copy(bytes.begin(), bytes.end(), descriptor.ptr<unsigned char>());
}

dbow2_vocabulary::dbow2_vocabulary(const std::string &filename)
{
    const dbow3_vocabulary binary(filename);
    const auto             parameters = binary.parameters();
    _descriptor_type                  = binary.descriptor_type();
    _descriptor_size                  = binary.descriptor_size();
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