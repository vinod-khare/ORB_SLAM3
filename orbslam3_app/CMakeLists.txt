cmake_minimum_required(VERSION 3.11)

project(orbslam3_app LANGUAGES C CXX)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(rapidcsv CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_executable(orbslam3_app
    source/orbslam3.cc
    source/folder_reader.cc)

target_compile_options(orbslam3_app PRIVATE -Wall)
target_include_directories(orbslam3_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(orbslam3_app PRIVATE ORB_SLAM3 Boost::program_options rapidcsv::rapidcsv spdlog::spdlog)
