find_package(Catch2 CONFIG REQUIRED)

add_executable(orbslam3_tests
    timestamp_test.cpp)

target_link_libraries(orbslam3_tests PRIVATE ORB_SLAM3 Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(orbslam3_tests)