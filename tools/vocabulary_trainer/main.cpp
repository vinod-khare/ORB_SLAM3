#include "keypoint_extractor.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <DBow3/Vocabulary.h>
#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>

namespace
{
namespace fs = std::filesystem;

bool is_image(const fs::path &path)
{
    auto extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".tif" || extension == ".tiff" || extension == ".pgm";
}
} // namespace

int main(int argc, char **argv)
{
    namespace po = boost::program_options;
    fs::path temporary;
    try
    {
        std::string             sequence;
        std::string             output;
        std::string             descriptor_name;
        std::string             detector_name;
        int                     branching{};
        int                     levels{};
        int                     stride{};
        int                     max_frames{};
        int                     features{};
        unsigned int            seed{};
        po::options_description options("Train a compressed binary DBoW vocabulary from an image directory (recursive, lexically sorted)");
        options.add_options()("help,h", "Show options")("sequence", po::value(&sequence)->required(), "Image directory, e.g. dso/cam0/images")(
            "output", po::value(&output)->required(), "New .dbow3 file; never overwritten")(
            "descriptor", po::value(&descriptor_name)->default_value("orb"), "Descriptor type; same names as keypoint.descriptor.type")(
            "detector", po::value(&detector_name)->default_value("fast"), "Detector type; same names as keypoint.detector.type")(
            "branching", po::value(&branching)->default_value(10), "Tree branching factor (2..20)")(
            "levels", po::value(&levels)->default_value(5), "Tree depth (1..10); maximum words = branching^levels")("stride", po::value(&stride)->default_value(10), "Sample every Nth image")(
            "max-frames", po::value(&max_frames)->default_value(500), "Maximum sampled images (bounds training memory)")(
            "features", po::value(&features)->default_value(1000), "Maximum descriptors per sampled image")("seed", po::value(&seed)->default_value(42), "OpenCV random seed");
        po::variables_map arguments;
        po::store(po::parse_command_line(argc, argv, options), arguments);
        if (arguments.contains("help"))
        {
            std::ostringstream usage;
            usage << options;
            spdlog::info("{}", usage.str());
            return 0;
        }
        po::notify(arguments);
        if (branching < 2 || branching > 20 || levels < 1 || levels > 10 || stride < 1 || max_frames < 2 || features < 1)
        {
            throw std::invalid_argument("Invalid training limits; use --help for ranges");
        }
        for (auto *name : {&descriptor_name, &detector_name})
        {
            std::ranges::transform(*name, name->begin(), [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        }
        const auto descriptor = magic_enum::enum_cast<ORB_SLAM3::keypoint_descriptor_type>(descriptor_name);
        const auto detector   = magic_enum::enum_cast<ORB_SLAM3::keypoint_detector_type>(detector_name);
        if (!descriptor || !detector)
        {
            throw std::invalid_argument("Unknown detector or descriptor type");
        }
        const fs::path destination(output);
        if (destination.extension() != ".dbow3" || fs::exists(destination))
        {
            throw std::invalid_argument("Output must be a new .dbow3 file");
        }
        std::vector<fs::path> images;
        for (const auto &entry : fs::recursive_directory_iterator(sequence))
        {
            if (entry.is_regular_file() && is_image(entry.path()))
            {
                images.push_back(entry.path());
            }
        }
        std::ranges::sort(images);
        if (images.empty())
        {
            throw std::runtime_error("No images found in sequence");
        }
        cv::theRNG().state = seed;
        ORB_SLAM3::keypoint_extractor extractor(features, 1.2f, 8, 20, 7, *detector, *descriptor);
        std::vector<cv::Mat>          training;
        std::size_t                   descriptor_count{};
        int                           sampled{};
        for (std::size_t index = 0; index < images.size() && sampled < max_frames; index += static_cast<std::size_t>(stride), ++sampled)
        {
            const auto image = cv::imread(images[index].string(), cv::IMREAD_GRAYSCALE);
            if (image.empty())
            {
                throw std::runtime_error("Cannot decode image: " + images[index].string());
            }
            std::vector<ORB_SLAM3::keypoint> keypoints;
            std::vector<int>                 overlap{0, 0};
            cv::Mat                          descriptors;
            extractor(image, cv::noArray(), keypoints, descriptors, overlap);
            if (!descriptors.empty())
            {
                training.push_back(descriptors.rowRange(0, std::min(features, descriptors.rows)).clone());
                descriptor_count += static_cast<std::size_t>(training.back().rows);
            }
        }
        if (training.size() < 2 || descriptor_count < static_cast<std::size_t>(branching))
        {
            throw std::runtime_error("Need at least two feature-bearing images and branching descriptors");
        }
        spdlog::info("Training {} vocabulary: {} images, {} descriptors, {}x{} tree", descriptor_name, training.size(), descriptor_count, branching, levels);
        if (*descriptor != ORB_SLAM3::keypoint_descriptor_type::orb)
        {
            spdlog::warn("The SLAM app uses an ORB place-recognition channel and cannot load this {} vocabulary", descriptor_name);
        }
        DBoW3::Vocabulary vocabulary(branching, levels, DBoW3::TF_IDF, DBoW3::L1_NORM);
        vocabulary.create(training);
        if (vocabulary.empty())
        {
            throw std::runtime_error("Training produced an empty vocabulary");
        }
        if (!destination.parent_path().empty())
        {
            fs::create_directories(destination.parent_path());
        }
        {
            const fs::path candidate(output + ".tmp");
            std::ofstream  stream(candidate, std::ios::binary | std::ios::noreplace);
            if (!stream)
            {
                throw std::runtime_error("Cannot create temporary output: " + candidate.string());
            }
            temporary = candidate;
            stream.exceptions(std::ios::badbit | std::ios::failbit);
            vocabulary.toStream(stream, true);
            stream.close();
        }
        DBoW3::Vocabulary restored;
        std::ifstream     stream(temporary, std::ios::binary);
        restored.fromStream(stream);
        if (!stream || restored.size() != vocabulary.size() || restored.getDescritorType() != vocabulary.getDescritorType() || restored.getDescritorSize() != vocabulary.getDescritorSize())
        {
            throw std::runtime_error("Binary vocabulary verification failed");
        }
        for (const auto &descriptors : training)
        {
            DBoW3::BowVector expected;
            DBoW3::BowVector actual;
            vocabulary.transform(descriptors, expected);
            restored.transform(descriptors, actual);
            if (expected != actual)
            {
                throw std::runtime_error("Binary vocabulary quantization changed after saving");
            }
        }
        fs::create_hard_link(temporary, destination);
        fs::remove(temporary);
        temporary.clear();
        spdlog::info("Saved {} words to {} ({} bytes)", restored.size(), output, fs::file_size(destination));
        return 0;
    }
    catch (const std::exception &error)
    {
        if (!temporary.empty())
        {
            std::error_code ignored;
            fs::remove(temporary, ignored);
        }
        spdlog::error("{}", error.what());
        return 1;
    }
}