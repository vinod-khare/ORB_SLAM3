#include "Settings.h"
#include "System.h"
#include "Tracking.h"

#include <chrono>
#include <filesystem>
#include <fstream>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <yaml-cpp/yaml.h>

void write_vocabulary_fixture(const std::string &filename, int scoring, int weighting, int descriptor_type, int descriptor_size);

namespace
{
class extraction_tracker : public ORB_SLAM3::Tracking
{
  public:
    using Tracking::cameras;
    using Tracking::create_frame;
    using Tracking::extract_current_features;
    using Tracking::Tracking;

    ORB_SLAM3::keypoint_extractor &extractor(bool initial) { return initial ? *initial_keypoint_extractor : *left_keypoint_extractor; }
};

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
        config = YAML::Load(R"(data:
      calibration:
        file: calibration/camchain.yaml
        imu_file: calibration/imu.yaml
Camera.fps: 20
Camera.RGB: 1
keypoint:
  detector:
    type: FAST
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

TEST_CASE("Tracking preserves initialization extraction during deferred recovery", "[tracking][features]")
{
    const auto   sensor = GENERATE(ORB_SLAM3::System::MONOCULAR, ORB_SLAM3::System::IMU_MONOCULAR);
    imu_settings files;
    files.write();
    ORB_SLAM3::Settings settings(files.path.string(), sensor);
    ORB_SLAM3::atlas    atlas(0);
    extraction_tracker  tracker(nullptr, nullptr, &atlas, nullptr, sensor, &settings, settings.cameras());
    REQUIRE(&tracker.cameras == &settings.cameras());
    tracker.mImGray = cv::Mat(512, 512, CV_8UC1);
    cv::RNG random(17);
    random.fill(tracker.mImGray, cv::RNG::UNIFORM, 0, 256);
    ORB_SLAM3::Frame::mbInitialComputations = true;

    const bool initial = GENERATE(false, true);
    tracker.create_frame(1.0, tracker.extractor(initial));
    const auto descriptors = descriptor_matrix(tracker.mCurrentFrame.mvKeys);
    REQUIRE(descriptors.rows > 0);
    REQUIRE((descriptors.rows > 250) == initial);
    REQUIRE(tracker.mCurrentFrame.mpPrevFrame == (sensor == ORB_SLAM3::System::IMU_MONOCULAR ? &tracker.mLastFrame : nullptr));
    tracker.mLastFrame    = tracker.mCurrentFrame;
    tracker.mCurrentFrame = ORB_SLAM3::Frame(tracker.mLastFrame, 1.05);
    REQUIRE(tracker.mCurrentFrame.flow_only);
    tracker.extract_current_features();
    REQUIRE_FALSE(tracker.mCurrentFrame.flow_only);
    REQUIRE(tracker.mCurrentFrame.mpPrevFrame == &tracker.mLastFrame);
    REQUIRE(tracker.mCurrentFrame.mTimeStamp == 1.05);
    REQUIRE(tracker.mCurrentFrame.N == descriptors.rows);
    REQUIRE(cv::norm(descriptor_matrix(tracker.mCurrentFrame.mvKeys), descriptors, cv::NORM_INF) == 0);
}

TEST_CASE("System joins workers on shutdown and destruction", "[statistics][shutdown]")
{
    imu_settings files;
    const auto   sensor      = GENERATE(ORB_SLAM3::System::MONOCULAR, ORB_SLAM3::System::STEREO);
    const bool   fisheye     = GENERATE(false, true);
    auto         calibration = YAML::LoadFile((files.directory / "calibration/camchain.yaml").string());
    if (fisheye)
    {
        calibration["cam0"]["distortion_model"]  = "equidistant";
        calibration["cam0"]["distortion_coeffs"] = std::vector<float>{0, 0, 0, 0};
    }
    if (sensor == ORB_SLAM3::System::STEREO)
    {
        calibration["cam1"]              = YAML::Clone(calibration["cam0"]);
        calibration["cam1"]["T_cn_cnm1"] = YAML::Load("[[1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]");
        files.config["Stereo.ThDepth"]   = 40.5;
    }
    std::ofstream(files.directory / "calibration/camchain.yaml") << calibration;
    files.write();
    const auto vocabulary_path = files.directory / "vocabulary.dbow3";
    write_vocabulary_fixture(vocabulary_path.string(), 0, 0, CV_8U, 32);

    const bool        explicit_shutdown = GENERATE(false, true);
    ORB_SLAM3::System system(vocabulary_path.string(), files.path.string(), sensor);
    REQUIRE(system.cameras().size() == (sensor == ORB_SLAM3::System::STEREO ? 2 : 1));
    REQUIRE(system.cameras()[0]->GetType() == (fisheye ? ORB_SLAM3::GeometricCamera::CAM_FISHEYE : ORB_SLAM3::GeometricCamera::CAM_PINHOLE));
    const auto   &primary_camera = *system.cameras()[0];
    const cv::Mat image(512, 512, CV_8UC1, cv::Scalar(0));
    if (sensor == ORB_SLAM3::System::STEREO)
    {
        system.TrackStereo(image, image, 1.0);
    }
    else
    {
        system.TrackMonocular(image, 1.0);
    }
    REQUIRE(&primary_camera == &*system.cameras()[0]);
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
        files.config["data"]["calibration"]["imu_file"] = (files.directory / "calibration/imu.yaml").string();
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

TEST_CASE("Tracking rejects a vocabulary that does not match the selected descriptor", "[settings][dbow]")
{
    imu_settings files;
    files.config["keypoint"]["descriptor"]["type"] = "SIFT";
    files.write();
    const auto vocabulary_path = files.directory / "orb-vocabulary.dbow3";
    write_vocabulary_fixture(vocabulary_path.string(), 0, 0, CV_8U, 32);
    REQUIRE_THROWS_WITH(ORB_SLAM3::System(vocabulary_path.string(), files.path.string(), ORB_SLAM3::System::MONOCULAR),
                        Catch::Matchers::ContainsSubstring("does not match keypoint.descriptor.type"));
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

TEST_CASE("Matcher setting uses tracking YAML and command-line precedence", "[settings][tracking]")
{
    imu_settings files;
    files.config["tracking"]["matcher"] = "flann";
    files.write();
    REQUIRE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR).matcher_backend() == ORB_SLAM3::descriptor_matcher_type::flann);

    ORB_SLAM3::feature_overrides overrides;
    overrides.matcher = "brute_force";
    REQUIRE(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR, overrides).matcher_backend() == ORB_SLAM3::descriptor_matcher_type::brute_force);

    files.config["tracking"]["matcher"] = "invalid";
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR), "Unknown tracking.matcher: invalid");
}

TEST_CASE("Contrib feature settings support YAML and command-line overrides", "[settings][features]")
{
    using namespace ORB_SLAM3;
    imu_settings      files;
    feature_overrides overrides;
    files.write();
    Settings settings(files.path.string(), System::MONOCULAR);
    REQUIRE(settings.nFeatures() == 100);
    REQUIRE(settings.scaleFactor() == 1.2f);
    REQUIRE(settings.nLevels() == 4);
    REQUIRE(settings.initThFAST() == 20);
    REQUIRE(settings.minThFAST() == 7);
    SECTION("Detectors")
    {
        using detector                               = keypoint_detector_type;
        const auto [name, type]                      = GENERATE(std::pair{"STAR", detector::star},
                                           std::pair{"HARRIS_LAPLACE", detector::harris_laplace},
                                           std::pair{"MSD", detector::msd},
                                           std::pair{"TBMR", detector::tbmr},
                                           std::pair{"SURF", detector::surf});
        files.config["keypoint"]["detector"]["type"] = name;
        files.write();
        REQUIRE(Settings(files.path.string(), System::MONOCULAR).keypoint_detector() == type);
        overrides.detector = "FAST";
        REQUIRE(Settings(files.path.string(), System::MONOCULAR, overrides).keypoint_detector() == detector::fast);
        files.config["keypoint"]["detector"]["type"] = "fast";
        files.write();
        overrides.detector = name;
        REQUIRE(Settings(files.path.string(), System::MONOCULAR, overrides).keypoint_detector() == type);
    }
    SECTION("Descriptors")
    {
        using descriptor                               = keypoint_descriptor_type;
        const auto [name, type]                        = GENERATE(std::pair{"BRIEF", descriptor::brief},
                                           std::pair{"FREAK", descriptor::freak},
                                           std::pair{"LATCH", descriptor::latch},
                                           std::pair{"BEBLID", descriptor::beblid},
                                           std::pair{"TEBLID", descriptor::teblid},
                                           std::pair{"DAISY", descriptor::daisy},
                                           std::pair{"VGG", descriptor::vgg},
                                           std::pair{"BOOSTDESC", descriptor::boostdesc},
                                           std::pair{"SURF", descriptor::surf});
        files.config["keypoint"]["descriptor"]["type"] = name;
        files.write();
        REQUIRE(Settings(files.path.string(), System::MONOCULAR).keypoint_descriptor() == type);
        overrides.descriptor = "ORB";
        REQUIRE(Settings(files.path.string(), System::MONOCULAR, overrides).keypoint_descriptor() == descriptor::orb);
        files.config["keypoint"]["descriptor"]["type"] = "orb";
        files.write();
        overrides.descriptor = name;
        REQUIRE(Settings(files.path.string(), System::MONOCULAR, overrides).keypoint_descriptor() == type);
    }
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
    files.config["data"]["calibration"].remove("imu_file");
    files.write();
    REQUIRE_NOTHROW(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::MONOCULAR));
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::IMU_MONOCULAR), Catch::Matchers::ContainsSubstring("data.calibration.imu_file"));

    files.config["data"]["calibration"]["imu_file"] = "missing.yaml";
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::Settings(files.path.string(), ORB_SLAM3::System::IMU_MONOCULAR), Catch::Matchers::ContainsSubstring("missing.yaml"));
}