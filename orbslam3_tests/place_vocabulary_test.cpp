#include "place_vocabulary.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <latch>
#include <thread>

#include <opencv2/core.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

namespace
{
struct vocabulary_file
{
    std::filesystem::path path = std::filesystem::temp_directory_path() / ("orbslam3_vocabulary_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    std::vector<cv::Mat>  features;

    vocabulary_file(int scoring = 0, int weighting = 0)
    {
        std::ofstream stream(path);
        stream << "2 6 " << scoring << ' ' << weighting << '\n';
        for (int id = 1; id < 127; ++id)
        {
            const bool leaf = id >= 63;
            stream << (id - 1) / 2 << ' ' << leaf;
            cv::Mat descriptor(1, 32, CV_8U);
            for (int column = 0; column < 32; ++column)
            {
                const int value = (id * 37 + column * 13) % 256;
                stream << ' ' << value;
                descriptor.at<unsigned char>(0, column) = static_cast<unsigned char>(value);
            }
            stream << ' ' << 0.25 * (id % 4 + 1) << '\n';
            if (leaf)
            {
                features.push_back(descriptor);
            }
        }
    }
    ~vocabulary_file() { std::filesystem::remove(path); }
};
} // namespace

TEST_CASE("DBoW backends preserve word IDs, feature nodes, scores and runtime switching", "[dbow]")
{
    const vocabulary_file       file;
    ORB_SLAM3::place_vocabulary vocabulary(file.path.string());
    REQUIRE(vocabulary.version() == 2);
    REQUIRE(vocabulary.size() == 64);
    DBoW2::BowVector     bow;
    DBoW2::FeatureVector nodes;
    vocabulary.transform(file.features, bow, nodes, 4);
    REQUIRE_FALSE(bow.empty());
    REQUIRE_FALSE(nodes.empty());
    const auto score = vocabulary.score(bow, bow);
    REQUIRE(score == Catch::Approx(1));
    vocabulary.set_version(3);
    REQUIRE(vocabulary.version() == 3);
    REQUIRE(vocabulary.size() == 64);
    DBoW2::BowVector     bow3;
    DBoW2::FeatureVector nodes3;
    vocabulary.transform(file.features, bow3, nodes3, 4);
    REQUIRE(bow3 == bow);
    REQUIRE(nodes3 == nodes);
    REQUIRE(vocabulary.score(bow, bow3) == Catch::Approx(score));
    vocabulary.transform({}, bow3, nodes3, 4);
    REQUIRE(bow3.empty());
    REQUIRE(nodes3.empty());
    vocabulary.set_version(2);
    vocabulary.transform(file.features, bow3, nodes3, 4);
    REQUIRE(bow3 == bow);
    REQUIRE(nodes3 == nodes);
    REQUIRE_THROWS_WITH(vocabulary.set_version(4), "dbow_version must be 2 or 3");
    REQUIRE(vocabulary.version() == 2);
}

TEST_CASE("Failed backend switches retain a usable vocabulary", "[dbow]")
{
    vocabulary_file             file;
    const int                   initial = GENERATE(2, 3);
    ORB_SLAM3::place_vocabulary vocabulary(file.path.string(), initial);
    std::filesystem::remove(file.path);
    REQUIRE_THROWS(vocabulary.set_version(5 - initial));
    REQUIRE(vocabulary.version() == initial);
    REQUIRE(vocabulary.size() == 64);
    DBoW2::BowVector     bow;
    DBoW2::FeatureVector nodes;
    REQUIRE_NOTHROW(vocabulary.transform(file.features, bow, nodes, 4));
    REQUIRE_FALSE(bow.empty());
}

TEST_CASE("DBoW backends agree for every text-vocabulary scoring and weighting mode", "[dbow]")
{
    const int scoring   = GENERATE(0, 1, 2, 3, 4, 5);
    const int weighting = GENERATE(0, 1, 2, 3);
    CAPTURE(scoring, weighting);
    const vocabulary_file       file(scoring, weighting);
    ORB_SLAM3::place_vocabulary vocabulary(file.path.string());
    const std::vector<cv::Mat>  subset(file.features.begin(), file.features.begin() + 16);
    DBoW2::BowVector            reference;
    DBoW2::BowVector            candidate;
    DBoW2::FeatureVector        reference_nodes;
    DBoW2::FeatureVector        candidate_nodes;
    vocabulary.transform(file.features, reference, reference_nodes, 4);
    vocabulary.transform(subset, candidate, candidate_nodes, 4);
    const auto expected_score = vocabulary.score(reference, candidate);
    vocabulary.set_version(3);
    DBoW2::BowVector     actual;
    DBoW2::FeatureVector actual_nodes;
    vocabulary.transform(subset, actual, actual_nodes, 4);
    REQUIRE(actual == candidate);
    REQUIRE(actual_nodes == candidate_nodes);
    REQUIRE(vocabulary.score(reference, actual) == Catch::Approx(expected_score).margin(1e-9));
}

TEST_CASE("Vocabulary queries remain valid while the backend changes", "[dbow]")
{
    const vocabulary_file       file;
    ORB_SLAM3::place_vocabulary vocabulary(file.path.string());
    DBoW2::BowVector            expected;
    DBoW2::FeatureVector        expected_nodes;
    vocabulary.transform(file.features, expected, expected_nodes, 4);
    std::atomic<bool> valid{true};
    std::atomic<int>  queries{0};
    std::latch        started{1};
    std::jthread      worker(
        [&](std::stop_token stop)
        {
            do
            {
                DBoW2::BowVector     bow;
                DBoW2::FeatureVector nodes;
                vocabulary.transform(file.features, bow, nodes, 4);
                if (bow != expected || nodes != expected_nodes || vocabulary.score(bow, expected) < 0.999999)
                {
                    valid = false;
                }
                if (++queries == 1)
                {
                    started.count_down();
                }
            } while (!stop.stop_requested());
        });
    started.wait();
    for (int i = 0; i < 8; ++i)
    {
        vocabulary.set_version(i % 2 + 2);
    }
    worker.request_stop();
    worker.join();
    REQUIRE(valid.load());
    REQUIRE(queries.load() > 0);
}

TEST_CASE("Full ORB text vocabulary preserves DBoW word and node identities", "[.][dbow-full]")
{
    const auto *filename = std::getenv("ORBSLAM3_TEST_VOCABULARY");
    if (!filename)
    {
        SKIP("Set ORBSLAM3_TEST_VOCABULARY to an ORBvoc.txt path");
    }
    ORB_SLAM3::place_vocabulary vocabulary(filename);
    const auto                  words = vocabulary.size();
    cv::Mat                     descriptors(256, 32, CV_8U);
    cv::RNG                     random(42);
    random.fill(descriptors, cv::RNG::UNIFORM, 0, 256);
    std::vector<cv::Mat> features;
    for (int row = 0; row < descriptors.rows; ++row)
    {
        features.push_back(descriptors.row(row));
    }
    DBoW2::BowVector     expected;
    DBoW2::FeatureVector expected_nodes;
    vocabulary.transform(features, expected, expected_nodes, 4);
    const std::vector<cv::Mat> subset(features.begin(), features.begin() + 32);
    DBoW2::BowVector           candidate;
    DBoW2::FeatureVector       candidate_nodes;
    vocabulary.transform(subset, candidate, candidate_nodes, 4);
    const auto score = vocabulary.score(expected, candidate);
    vocabulary.set_version(3);
    REQUIRE(vocabulary.size() == words);
    DBoW2::BowVector     actual;
    DBoW2::FeatureVector actual_nodes;
    vocabulary.transform(features, actual, actual_nodes, 4);
    REQUIRE(actual_nodes == expected_nodes);
    REQUIRE(actual.size() == expected.size());
    for (const auto &[word, weight] : expected)
    {
        REQUIRE(actual.contains(word));
        REQUIRE(actual.at(word) == Catch::Approx(weight).epsilon(1e-6));
    }
    REQUIRE(vocabulary.score(actual, candidate) == Catch::Approx(score).epsilon(1e-6));
}
