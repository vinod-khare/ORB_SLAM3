cmake_minimum_required(VERSION 3.25)

project(ORB_SLAM3 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(orbslam3_lib)
add_subdirectory(orbslam3_app)

include(CTest)
if(BUILD_TESTING)
	add_subdirectory(orbslam3_tests)
endif()
