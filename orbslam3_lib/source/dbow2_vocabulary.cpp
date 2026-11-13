#include "dbow2_vocabulary.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ORB_SLAM3
{
namespace
{
template <typename VALUE_T_> VALUE_T_ read_binary(std::istream &stream)
{
    VALUE_T_ value{};

    if (!stream.read(reinterpret_cast<char *>(&value), sizeof(value)))
    {
        throw std::runtime_error("Truncated binary vocabulary");
    }

    return value;
}

// DBoW3 stores independent QuickLZ level-1 blocks of at most 10,000 bytes.
std::string read_vocabulary_block(std::istream &stream)
{
    const auto flags = read_binary<std::uint8_t>(stream);

    if ((flags & 0xfc) != 0x44)
    {
        throw std::runtime_error("Unsupported binary vocabulary compression");
    }

    const auto read_size       = [&]() -> std::uint32_t { return flags & 2 ? read_binary<std::uint32_t>(stream) : read_binary<std::uint8_t>(stream); };
    const auto compressed_size = read_size();
    const auto size            = read_size();
    const auto header_size     = flags & 2 ? 9u : 3u;

    if (size == 0 || size > 10000 || compressed_size < header_size || compressed_size > 10400)
    {
        throw std::runtime_error("Invalid binary vocabulary compression block");
    }

    std::string input(compressed_size - header_size, '\0');

    if (!stream.read(input.data(), static_cast<std::streamsize>(input.size())))
    {
        throw std::runtime_error("Truncated binary vocabulary compression block");
    }

    if (!(flags & 1))
    {
        if (input.size() != size)
        {
            throw std::runtime_error("Invalid uncompressed vocabulary block");
        }

        return input;
    }

    std::string output;
    output.reserve(size);

    std::array<int, 4096> offsets;
    offsets.fill(-1);

    int           last_hashed = -1;
    std::size_t   position{};
    std::uint32_t control{1};

    const auto take = [&]() -> std::uint32_t
    {
        if (position == input.size())
        {
            throw std::runtime_error("Truncated binary vocabulary compression token");
        }

        return static_cast<unsigned char>(input[position++]);
    };

    const auto update_hashes = [&](int end)
    {
        while (last_hashed < end)
        {
            const auto index = static_cast<std::size_t>(++last_hashed);
            const auto value = static_cast<unsigned char>(output[index]) | (static_cast<std::uint32_t>(static_cast<unsigned char>(output[index + 1])) << 8) |
                               (static_cast<std::uint32_t>(static_cast<unsigned char>(output[index + 2])) << 16);

            offsets[((value >> 12) ^ value) & 4095] = last_hashed;
        }
    };

    while (output.size() < size)
    {
        if (control == 1)
        {
            control = 0;

            for (int byte = 0; byte < 4; ++byte)
            {
                control |= take() << (8 * byte);
            }

            if (!(control & 0x80000000u))
            {
                throw std::runtime_error("Invalid binary vocabulary compression control");
            }
        }

        if (control & 1)
        {
            const auto low    = take();
            const auto token  = low | (take() << 8);
            const auto length = low & 15 ? (low & 15) + 2 : take();
            const auto offset = offsets[token >> 4];
            const auto start  = output.size();

            if (offset < 0 || static_cast<std::size_t>(offset) >= start || length < 3 || length > size - start)
            {
                throw std::runtime_error("Invalid binary vocabulary compression match");
            }

            for (std::size_t byte = 0; byte < length; ++byte)
            {
                output.push_back(output[static_cast<std::size_t>(offset) + byte]);
            }

            update_hashes(static_cast<int>(start));
            last_hashed = static_cast<int>(output.size()) - 1;
        }
        else
        {
            output.push_back(static_cast<char>(take()));
            update_hashes(static_cast<int>(output.size()) - 3);
        }

        control >>= 1;
    }

    return output;
}
} // namespace

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
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Cannot open binary vocabulary: " + filename);
    }

    if (read_binary<std::uint64_t>(file) != 88877711233ULL)
    {
        throw std::runtime_error("Invalid binary vocabulary signature: " + filename);
    }

    const auto compressed = read_binary<std::uint8_t>(file);
    const auto node_count = read_binary<std::uint32_t>(file);

    if (compressed > 1 || node_count < 2)
    {
        throw std::runtime_error("Invalid binary vocabulary header: " + filename);
    }

    std::stringstream payload;

    if (compressed)
    {
        const auto chunks = read_binary<std::uint32_t>(file);

        if (chunks == 0)
        {
            throw std::runtime_error("Empty binary vocabulary: " + filename);
        }

        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk)
        {
            payload << read_vocabulary_block(file);
        }
    }

    std::istream &stream = compressed ? static_cast<std::istream &>(payload) : file;

    const auto payload_start = stream.tellg();
    stream.seekg(0, std::ios::end);

    const auto payload_size = stream.tellg() - payload_start;
    stream.seekg(payload_start);

    if (payload_size < 0 || node_count > static_cast<std::uint64_t>(payload_size) / 28)
    {
        throw std::runtime_error("Invalid binary vocabulary node count: " + filename);
    }

    m_k                  = read_binary<int>(stream);
    m_L                  = read_binary<int>(stream);
    const auto scoring   = read_binary<int>(stream);
    const auto weighting = read_binary<int>(stream);

    if (m_k < 1 || m_L < 1 || scoring < 0 || scoring > 5 || weighting < 0 || weighting > 3)
    {
        throw std::runtime_error("Invalid binary vocabulary parameters: " + filename);
    }

    m_scoring   = static_cast<DBoW2::ScoringType>(scoring);
    m_weighting = static_cast<DBoW2::WeightingType>(weighting);

    createScoringObject();

    m_nodes.resize(node_count);
    std::vector<bool> seen(node_count);
    seen[0] = true;

    for (std::uint32_t index = 1; index < node_count; ++index)
    {
        const auto id     = read_binary<std::uint32_t>(stream);
        const auto parent = read_binary<std::uint32_t>(stream);

        if (id == 0 || id >= node_count || seen[id] || parent >= node_count || !seen[parent])
        {
            throw std::runtime_error("Invalid binary vocabulary node: " + filename);
        }

        seen[id]           = true;
        auto &node         = m_nodes[id];
        node.id            = id;
        node.parent        = parent;
        node.weight        = read_binary<double>(stream);
        const auto columns = read_binary<int>(stream);
        const auto rows    = read_binary<int>(stream);
        const auto type    = read_binary<int>(stream);

        if (rows != 1 || columns <= 0 || (type != CV_8U && type != CV_32F) || static_cast<std::uint64_t>(columns) * (type == CV_32F ? 4u : 1u) > static_cast<std::uint64_t>(payload_size) ||
            (index > 1 && (columns != _descriptor_size || type != _descriptor_type)))
        {
            throw std::runtime_error("Invalid binary vocabulary descriptor: " + filename);
        }

        _descriptor_type = type;
        _descriptor_size = columns;

        node.descriptor.create(1, columns, type);

        if (!stream.read(reinterpret_cast<char *>(node.descriptor.data), static_cast<std::streamsize>(node.descriptor.total() * node.descriptor.elemSize())))
        {
            throw std::runtime_error("Truncated binary vocabulary descriptor: " + filename);
        }

        m_nodes[parent].children.push_back(id);
    }

    const auto word_count = read_binary<std::uint32_t>(stream);

    if (word_count == 0 || word_count >= node_count)
    {
        throw std::runtime_error("Invalid binary vocabulary word count: " + filename);
    }

    m_words.resize(word_count);

    for (std::uint32_t index = 0; index < word_count; ++index)
    {
        const auto word = read_binary<std::uint32_t>(stream);
        const auto id   = read_binary<std::uint32_t>(stream);

        if (word >= word_count || m_words[word] || id == 0 || id >= node_count || !m_nodes[id].isLeaf())
        {
            throw std::runtime_error("Invalid binary vocabulary word ID: " + filename);
        }

        m_nodes[id].word_id = word;
        m_words[word]       = &m_nodes[id];
    }

    for (const auto &node : m_nodes)
    {
        if (node.isLeaf() && (node.word_id >= word_count || m_words[node.word_id] != &node))
        {
            throw std::runtime_error("Missing binary vocabulary word: " + filename);
        }
    }
}
} // namespace ORB_SLAM3