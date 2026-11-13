#include "settings.h"
#include "system.h"
#include "tracking.h"

#include <chrono>
#include <filesystem>
#include <fstream>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <yaml-cpp/yaml.h>

void write_vocabulary_fixture(const std::string &filename, int scoring, int weighting, int descriptor_type, int descriptor_size, int levels = 6);

namespace
{
class extraction_tracker : public ORB_SLAM3::tracking
{
  public:
    using tracking::cameras;
    using tracking::create_frame;
    using tracking::extract_current_features;
    using tracking::tracking;

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
    const auto   sensor = GENERATE(ORB_SLAM3::system::monocular, ORB_SLAM3::system::imu_monocular);
    imu_settings files;
    files.write();
    ORB_SLAM3::settings settings(files.path.string(), sensor);
    ORB_SLAM3::atlas    atlas(0);
    extraction_tracker  tracker(nullptr, nullptr, &atlas, nullptr, sensor, &settings, settings.cameras());
    REQUIRE(&tracker.cameras == &settings.cameras());
    tracker.gray_image = cv::Mat(512, 512, CV_8UC1);
    cv::RNG random(17);
    random.fill(tracker.gray_image, cv::RNG::UNIFORM, 0, 256);

    const bool initial = GENERATE(false, true);
    tracker.create_frame(1.0, tracker.extractor(initial));
    const auto descriptors = descriptor_matrix(tracker.current_frame.keypoints);
    REQUIRE(descriptors.rows > 0);
    REQUIRE((descriptors.rows > 250) == initial);
    REQUIRE(tracker.current_frame.previous_frame == (sensor == ORB_SLAM3::system::imu_monocular ? &tracker.last_frame : nullptr));
    tracker.last_frame    = tracker.current_frame;
    tracker.current_frame = ORB_SLAM3::frame(tracker.last_frame, 1.05, *tracker.cameras[0]);
    REQUIRE(tracker.current_frame.flow_only);
    tracker.extract_current_features();
    REQUIRE_FALSE(tracker.current_frame.flow_only);
    REQUIRE(tracker.current_frame.previous_frame == &tracker.last_frame);
    REQUIRE(tracker.current_frame.timestamp == 1.05);
    REQUIRE(tracker.current_frame.feature_count == descriptors.rows);
    REQUIRE(cv::norm(descriptor_matrix(tracker.current_frame.keypoints), descriptors, cv::NORM_INF) == 0);
}

TEST_CASE("System joins workers on shutdown and destruction", "[statistics][shutdown]")
{
    imu_settings files;
    const auto   sensor      = GENERATE(ORB_SLAM3::system::monocular, ORB_SLAM3::system::stereo);
    const bool   fisheye     = GENERATE(false, true);
    auto         calibration = YAML::LoadFile((files.directory / "calibration/camchain.yaml").string());
    if (fisheye)
    {
        calibration["cam0"]["distortion_model"]  = "equidistant";
        calibration["cam0"]["distortion_coeffs"] = std::vector<float>{0, 0, 0, 0};
    }
    if (sensor == ORB_SLAM3::system::stereo)
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
    auto              settings          = std::make_unique<ORB_SLAM3::settings>(files.path.string(), sensor);
    const auto       &cameras           = settings->cameras();
    ORB_SLAM3::system system(vocabulary_path.string(), std::move(settings));
    REQUIRE(cameras.size() == (sensor == ORB_SLAM3::system::stereo ? 2 : 1));
    REQUIRE(cameras[0]->get_type() == (fisheye ? ORB_SLAM3::geometric_camera::cam_fisheye : ORB_SLAM3::geometric_camera::cam_pinhole));
    if (sensor == ORB_SLAM3::system::stereo)
    {
        REQUIRE(cameras[0]->baseline() == Catch::Approx(0.1f));
        REQUIRE(cameras[0]->baseline_fx() == Catch::Approx(0.1f * cameras[0]->get_parameter(0)));
        REQUIRE(cameras[0]->left_to_right_pose().translation().x() == Catch::Approx(fisheye ? 0.1f : 0.f));
    }
    const auto   &primary_camera = *cameras[0];
    const cv::Mat image(512, 512, CV_8UC1, cv::Scalar(0));
    if (sensor == ORB_SLAM3::system::stereo)
    {
        system.track_stereo(image, image, 1.0);
    }
    else
    {
        system.track_monocular(image, 1.0);
    }
    REQUIRE(&primary_camera == &*cameras[0]);
    system.record_frame_statistics(2.0, 3.0);
    REQUIRE(system.statistics().get_snapshot().frames == 1);
    REQUIRE(system.statistics().get_snapshot().frame.frontend_ms == 3.0);
    if (explicit_shutdown)
    {
        system.shutdown();
        REQUIRE(system.is_shut_down());
        system.shutdown();
        REQUIRE(system.statistics().get_snapshot().frames == 1);
    }
}

TEST_CASE("Settings attach distortion to the primary camera", "[camera_calibration]")
{
    imu_settings files;
    auto         calibration                 = YAML::LoadFile((files.directory / "calibration/camchain.yaml").string());
    calibration["cam0"]["distortion_model"]  = "radtan";
    calibration["cam0"]["distortion_coeffs"] = std::vector<float>{-0.1f, 0, 0, 0};
    std::ofstream(files.directory / "calibration/camchain.yaml") << calibration;
    files.write();
    ORB_SLAM3::settings settings(files.path.string(), ORB_SLAM3::system::monocular);
    REQUIRE(settings.cameras()[0]->distortion_coefficients().at<float>(0) == Catch::Approx(-0.1f));
}

TEST_CASE("System exposes effective feature names before processing frames", "[settings][features]")
{
    imu_settings files;
    const auto   selection = GENERATE(0, 1, 2);
    if (selection != 0)
    {
        files.config["keypoint"]["detector"]["type"]   = "AgAsT";
        files.config["keypoint"]["descriptor"]["type"] = "BRIEF";
    }
    else
    {
        files.config["keypoint"]["detector"].remove("type");
    }
    files.write();
    ORB_SLAM3::feature_overrides overrides;
    if (selection == 2)
    {
        overrides.detector   = "GFTT";
        overrides.descriptor = "oRb";
    }
    const auto vocabulary_path = files.directory / "vocabulary.dbow3";
    write_vocabulary_fixture(vocabulary_path.string(), 0, 0, CV_8U, 32);
    auto                    settings = std::make_unique<ORB_SLAM3::settings>(files.path.string(), ORB_SLAM3::system::monocular, overrides);
    const ORB_SLAM3::system system(vocabulary_path.string(), std::move(settings));
    REQUIRE(system.keypoint_detector_name() == (selection == 0 ? "fast" : selection == 1 ? "agast" : "gftt"));
    REQUIRE(system.keypoint_descriptor_name() == (selection == 1 ? "brief" : "orb"));
    REQUIRE(system.statistics().get_snapshot().frames == 0);
}

TEST_CASE("System uses preconstructed settings without reopening configuration files", "[settings][system]")
{
    imu_settings files;
    const int    loop_closing = GENERATE(-1, 0, 1, 2);
    if (loop_closing >= 0)
    {
        files.config["loopClosing"] = loop_closing;
    }
    files.write();
    const auto vocabulary_path = files.directory / "vocabulary.dbow3";
    write_vocabulary_fixture(vocabulary_path.string(), 0, 0, CV_8U, 32);

    auto settings = std::make_unique<ORB_SLAM3::settings>(files.path.string(), ORB_SLAM3::system::monocular);
    REQUIRE(settings->sensor() == ORB_SLAM3::system::monocular);
    REQUIRE(settings->loop_closing_enabled() == (loop_closing != 0));
    const auto &cameras = settings->cameras();
    const auto *camera  = cameras[0].get();
    REQUIRE(std::filesystem::remove(files.path));
    REQUIRE(std::filesystem::remove(files.directory / "calibration/camchain.yaml"));

    ORB_SLAM3::system system(vocabulary_path.string(), std::move(settings));
    REQUIRE(cameras[0].get() == camera);
    const cv::Mat image(512, 512, CV_8UC1, cv::Scalar(0));
    REQUIRE_NOTHROW(system.track_monocular(image, 1.0));
    system.shutdown();
}

TEST_CASE("Settings and system reject invalid construction inputs", "[settings][system]")
{
    REQUIRE_THROWS_WITH(ORB_SLAM3::system("", nullptr), "System settings must not be null");
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings("", static_cast<ORB_SLAM3::sensor_type>(-1)), "Invalid sensor type");
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings("", static_cast<ORB_SLAM3::sensor_type>(6)), "Invalid sensor type");
}

TEST_CASE("IMU settings come from Kalibr calibration", "[settings][imu]")
{
    imu_settings files;
    if (GENERATE(false, true))
    {
        files.config["data"]["calibration"]["imu_file"] = (files.directory / "calibration/imu.yaml").string();
    }
    files.write();
    ORB_SLAM3::settings settings(files.path.string(), ORB_SLAM3::system::imu_monocular);
    REQUIRE(settings.gyroscope_noise() == Catch::Approx(0.00016));
    REQUIRE(settings.accelerometer_noise() == Catch::Approx(0.0028));
    REQUIRE(settings.gyroscope_random_walk() == Catch::Approx(0.000022));
    REQUIRE(settings.accelerometer_random_walk() == Catch::Approx(0.00086));
    REQUIRE(settings.imu_frequency() == Catch::Approx(200));
    REQUIRE(settings.imu_to_camera_pose().matrix().isApprox(Eigen::Matrix4f::Identity()));
}

TEST_CASE("Library settings select and validate DBoW independently of the app", "[settings][dbow]")
{
    imu_settings files;
    files.write();
    REQUIRE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular).dbow_version() == 2);
    files.config["dbow_version"] = 3;
    files.write();
    REQUIRE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular).dbow_version() == 3);
    ORB_SLAM3::feature_overrides overrides;
    overrides.dbow_version = 2;
    REQUIRE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular, overrides).dbow_version() == 2);
    files.config["dbow_version"] = 4;
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular), "dbow_version must be 2 or 3");
    overrides.dbow_version = 0;
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular, overrides), "dbow_version must be 2 or 3");
}

TEST_CASE("Tracking rejects a vocabulary that does not match the selected descriptor", "[settings][dbow]")
{
    imu_settings files;
    files.config["keypoint"]["descriptor"]["type"] = "SIFT";
    files.write();
    const auto vocabulary_path = files.directory / "orb-vocabulary.dbow3";
    write_vocabulary_fixture(vocabulary_path.string(), 0, 0, CV_8U, 32);
    auto settings = std::make_unique<ORB_SLAM3::settings>(files.path.string(), ORB_SLAM3::system::monocular);
    REQUIRE_THROWS_WITH(ORB_SLAM3::system(vocabulary_path.string(), std::move(settings)), Catch::Matchers::ContainsSubstring("does not match keypoint.descriptor.type"));
}

TEST_CASE("Optical-flow setting supports nested YAML and CLI precedence", "[settings][tracking]")
{
    imu_settings files;
    files.write();
    REQUIRE_FALSE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular).use_optical_flow());

    files.config["tracking"]["use_optical_flow"] = 1;
    files.write();
    REQUIRE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular).use_optical_flow());

    ORB_SLAM3::feature_overrides overrides;
    overrides.use_optical_flow = false;
    REQUIRE_FALSE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular, overrides).use_optical_flow());
    overrides.use_optical_flow                   = true;
    files.config["tracking"]["use_optical_flow"] = 0;
    files.write();
    REQUIRE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular, overrides).use_optical_flow());

    files.config["tracking"]["use_optical_flow"] = 2;
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular), "tracking.use_optical_flow must be 0 or 1");
}

TEST_CASE("Matcher setting uses tracking YAML and command-line precedence", "[settings][tracking]")
{
    imu_settings files;
    files.config["tracking"]["matcher"] = "flann";
    files.write();
    REQUIRE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular).matcher_backend() == ORB_SLAM3::descriptor_matcher_type::flann);

    ORB_SLAM3::feature_overrides overrides;
    overrides.matcher = "brute_force";
    REQUIRE(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular, overrides).matcher_backend() == ORB_SLAM3::descriptor_matcher_type::brute_force);

    files.config["tracking"]["matcher"] = "invalid";
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular), "Unknown tracking.matcher: invalid");
}

TEST_CASE("Contrib feature settings support YAML and command-line overrides", "[settings][features]")
{
    using namespace ORB_SLAM3;
    imu_settings      files;
    feature_overrides overrides;
    files.write();
    settings settings(files.path.string(), system::monocular);
    REQUIRE(settings.feature_count() == 100);
    REQUIRE(settings.scale_factor() == 1.2f);
    REQUIRE(settings.level_count() == 4);
    REQUIRE(settings.initial_fast_threshold() == 20);
    REQUIRE(settings.minimum_fast_threshold() == 7);
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
        REQUIRE(ORB_SLAM3::settings(files.path.string(), system::monocular).keypoint_detector() == type);
        overrides.detector = "FAST";
        REQUIRE(ORB_SLAM3::settings(files.path.string(), system::monocular, overrides).keypoint_detector() == detector::fast);
        files.config["keypoint"]["detector"]["type"] = "fast";
        files.write();
        overrides.detector = name;
        REQUIRE(ORB_SLAM3::settings(files.path.string(), system::monocular, overrides).keypoint_detector() == type);
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
        REQUIRE(ORB_SLAM3::settings(files.path.string(), system::monocular).keypoint_descriptor() == type);
        overrides.descriptor = "ORB";
        REQUIRE(ORB_SLAM3::settings(files.path.string(), system::monocular, overrides).keypoint_descriptor() == descriptor::orb);
        files.config["keypoint"]["descriptor"]["type"] = "orb";
        files.write();
        overrides.descriptor = name;
        REQUIRE(ORB_SLAM3::settings(files.path.string(), system::monocular, overrides).keypoint_descriptor() == type);
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
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::imu_monocular), Catch::Matchers::ContainsSubstring(key));
}

TEST_CASE("Legacy IMU settings are rejected", "[settings][imu]")
{
    imu_settings files;
    const auto   key  = GENERATE("IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk", "IMU.Frequency");
    files.config[key] = 1.0;
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::imu_monocular), Catch::Matchers::ContainsSubstring("no longer supported"));
}

TEST_CASE("Only inertial modes require an IMU calibration file", "[settings][imu]")
{
    imu_settings files;
    files.config["data"]["calibration"].remove("imu_file");
    files.write();
    REQUIRE_NOTHROW(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::monocular));
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::imu_monocular), Catch::Matchers::ContainsSubstring("data.calibration.imu_file"));

    files.config["data"]["calibration"]["imu_file"] = "missing.yaml";
    files.write();
    REQUIRE_THROWS_WITH(ORB_SLAM3::settings(files.path.string(), ORB_SLAM3::system::imu_monocular), Catch::Matchers::ContainsSubstring("missing.yaml"));
}