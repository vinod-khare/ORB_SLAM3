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
    imu_settings_test.cpp
    pnp_test.cpp
    ../orbslam3_app/source/command_line_options.cpp)

target_link_libraries(orbslam3_tests PRIVATE orbslam3_input orbslam3_ui Catch2::Catch2WithMain Boost::program_options spdlog::spdlog yaml-cpp::yaml-cpp)

include(Catch)
catch_discover_tests(orbslam3_tests)

option(ORBSLAM3_REGRESSION_TEST "Register the end-to-end TUM-VI accuracy regression test (ctest -L regression)" OFF)
if(ORBSLAM3_REGRESSION_TEST)
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME regression_stereo_corridor1
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/regression/regression.py
            --app $<TARGET_FILE:orbslam3_app>
            --slam-type stereo
            --output-folder ${CMAKE_BINARY_DIR}/regression/stereo)
    set_tests_properties(regression_stereo_corridor1 PROPERTIES LABELS regression TIMEOUT 1800)
endif()