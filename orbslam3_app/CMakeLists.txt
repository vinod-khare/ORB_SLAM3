cmake_minimum_required(VERSION 3.11)

project(orbslam3_app LANGUAGES C CXX)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(rapidcsv CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(orbslam3_input STATIC
    source/folder_reader.cc
    source/frame_prefetcher.cc)
target_compile_features(orbslam3_input PUBLIC cxx_std_20)
target_compile_options(orbslam3_input PRIVATE -Wall)
target_include_directories(orbslam3_input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(orbslam3_input PUBLIC ORB_SLAM3 Threads::Threads PRIVATE rapidcsv::rapidcsv spdlog::spdlog)

add_executable(orbslam3_app source/orbslam3.cc)

target_compile_options(orbslam3_app PRIVATE -Wall)
target_link_libraries(orbslam3_app PRIVATE orbslam3_input Boost::program_options spdlog::spdlog)
