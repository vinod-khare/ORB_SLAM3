find_package(Catch2 CONFIG REQUIRED)

add_executable(orbslam3_tests
    timestamp_test.cpp
    optical_flow_test.cpp)

target_link_libraries(orbslam3_tests PRIVATE ORB_SLAM3 Catch2::Catch2WithMain)

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