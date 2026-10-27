#include "Settings.h"
#include "System.h"

#include <chrono>
#include <filesystem>
#include <fstream>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <yaml-cpp/yaml.h>

namespace
{
struct imu_settings
{
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("orbslam3_imu_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path path      = directory / "settings.yaml";
    YAML::Node            config;
    YAML::Node            imu;

    imu_settings()
    {
        std::filesystem::create_directories(directory / "calibration");
        std::ofstream(directory / "calibration/camchain.yaml") << R"(cam0:
  camera_model: pinhole
  distortion_model: none
  intrinsics: [300, 300, 256, 256]
  resolution: [512, 512]
  T_cam_imu: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)";
        config = YAML::Load(R"(calibration:
  file: calibration/camchain.yaml
  imu_file: calibration/imu.yaml
Camera.fps: 20
Camera.RGB: 1
keypoint_extractor:
  n_features: 100
  scale_factor: 1.2
  n_levels: 4
  initial_threshold: 20
  minimum_threshold: 7
)");
        imu    = YAML::Load(R"(gyroscope_noise_density: 0.00016
accelerometer_noise_density: 0.0028
gyroscope_random_walk: 0.000022
accelerometer_random_walk: 0.00086
update_rate: 200.0
)");
    }

    ~imu_settings() { std::filesystem::remove_all(directory); }

    void write() const
    {
        std::ofstream(path) << "%YAML:1.0\n---\n" << config;
        std::ofstream(directory / "calibration/imu.yaml") << imu;
    }
};
} // namespace

TEST_CASE("System joins workers on shutdown and destruction", "[statistics][shutdown]")
{
    imu_settings files;
    files.write();
    const auto vocabulary_path = files.directory / "vocabulary.txt";
    {
        std::ofstream vocabulary(vocabulary_path);
        vocabulary << "2 1 0 0\n";
        for (int word = 0; word < 2; ++word)
        {
            vocabulary << "0 1";
            for (int column = 0; column < 32; ++column)
            {
                vocabulary << ' ' << word * 255;
            }
            vocabulary << " 1\n";
        }
    }

    const bool        explicit_shutdown = GENERATE(false, true);
    ORB_SLAM3::System system(vocabulary_path.string(), files.path.string(), ORB_SLAM3::System::MONOCULAR);
    const cv::Mat     image(512, 512, CV_8UC1, cv::Scalar(0));
    system.TrackMonocular(image, 1.0);
    system.record_frame_statistics(2.0, 3.0);
    REQUIRE(system.statistics().get_snapshot().frames == 1);
    REQUIRE(system.statistics().get_snapshot().frame.frontend_ms == 3.0);
    if (explicit_shutdown)
    {
        system.Shutdown();
        REQUIRE(system.isShutDown());
        system.Shutdown();
        REQUIRE(system.statistics().get_snapshot().frames == 1);
    }
}

TEST_CASE("IMU settings come from Kalibr calibration", "[settings][imu]")
{
    imu_settings files;
    if (GENERATE(false, true))
    {
        files.config["calibration"]["imu_file"] = (files.directory / "calibration/imu.yaml").string();
    }
    files.write();
    ORB_SLAM3::Settings settings(files.path.string(), ORB_SLAM3::System::IMU_MONOCULAR);
    REQUIRE(settings.noiseGyro() == Catch::Approx(0.00016));
    REQUIRE(settings.noiseAcc() == Catch::Approx(0.0028));
    REQUIRE(settings.gyroWalk() == Catch::Approx(0.000022));
    REQUIRE(settings.accWalk() == Catch::Approx(0.00086));
    REQUIRE(settings.imuFrequency() == Catch::Approx(200));
    REQUIRE(settings.Tbc().matrix().isApprox(Eigen::Matrix4f::Identity()));
}

TEST_CASE("Library settings select and validate DBoW independently of the app", "[settings][dbow]")
{
    imu_settings files;
    files.write();
    REQUIRE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR).dbow_version() == 2);
    files.config["dbow_version"] = 3;
    files.write();
    REQUIRE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR).dbow_version() == 3);
    ORB_SLAM3::feature_overrides overrides;
    overrides.dbow_version = 2;
    REQUIRE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR, overrides).dbow_version() == 2);
    files.config["dbow_version"] = 4;
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR), "dbow_version must be 2 or 3");
    overrides.dbow_version = 0;
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR, overrides), "dbow_version must be 2 or 3");
}

TEST_CASE("Optical-flow setting supports nested YAML and CLI precedence", "[settings][tracking]")
{
    imu_settings files;
    files.write();
    REQUIRE_FALSE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR).use_optical_flow());

    files.config["tracking"]["use_optical_flow"] = 1;
    files.write();
    REQUIRE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR).use_optical_flow());

    ORB_SLAM3::feature_overrides overrides;
    overrides.use_optical_flow = false;
    REQUIRE_FALSE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR, overrides).use_optical_flow());
    overrides.use_optical_flow                   = true;
    files.config["tracking"]["use_optical_flow"] = 0;
    files.write();
    REQUIRE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR, overrides).use_optical_flow());

    files.config["tracking"]["use_optical_flow"] = 2;
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR), "tracking.use_optical_flow must be 0 or 1");
}

TEST_CASE("IMU calibration values must be present and positive finite numbers", "[settings][imu]")
{
    imu_settings files;
    const auto   key   = GENERATE("gyroscope_noise_density", "accelerometer_noise_density", "gyroscope_random_walk", "accelerometer_random_walk", "update_rate");
    const auto   value = GENERATE("missing", "invalid", "0", "-1", ".nan", ".inf");
    CAPTURE(key, value);
    if (std::string(value) == "missing")
    {
        files.imu.remove(key);
    }
    else
    {
        files.imu[key] = YAML::Load(value);
    }
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::IMU_MONOCULAR), Catch::Matchers::ContainsSubstring(key));
}

TEST_CASE("Legacy IMU settings are rejected", "[settings][imu]")
{
    imu_settings files;
    const auto   key  = GENERATE("IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk", "IMU.Frequency");
    files.config[key] = 1.0;
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::IMU_MONOCULAR), Catch::Matchers::ContainsSubstring("no longer supported"));
}

TEST_CASE("Only inertial modes require an IMU calibration file", "[settings][imu]")
{
    imu_settings files;
    files.config["calibration"].remove("imu_file");
    files.write();
    REQUIRE_NOTHROW(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR));
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::IMU_MONOCULAR), Catch::Matchers::ContainsSubstring("calibration.imu_file"));

    files.config["calibration"]["imu_file"] = "missing.yaml";
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::IMU_MONOCULAR), Catch::Matchers::ContainsSubstring("missing.yaml"));
}