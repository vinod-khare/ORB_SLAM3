find_package(Catch2 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(spdlog CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(orbslam3_tests
    timestamp_test.cpp
    optical_flow_test.cpp
    frame_drawer_test.cpp
    ground_truth_test.cpp
    native_keypoints_test.cpp
    frame_prefetcher_test.cpp
    logging_options_test.cpp
    place_vocabulary_test.cpp
    imu_settings_test.cpp
    pnp_test.cpp
    ../orbslam3_app/source/command_line_options.cpp)

target_link_libraries(orbslam3_tests PRIVATE orbslam3_input orbslam3_ui Catch2::Catch2WithMain Boost::program_options spdlog::spdlog yaml-cpp::yaml-cpp)

include(Catch)
catch_discover_tests(orbslam3_tests)
