cmake_minimum_required(VERSION 3.11)

project(orbslam3_app LANGUAGES C CXX)

add_subdirectory(orbslam3_lib)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(rapidcsv CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_executable(orbslam3
        Examples/Monocular/orbslam3.cc
        Examples/Monocular/folder_reader.cc)
target_include_directories(orbslam3 PRIVATE ${PROJECT_SOURCE_DIR}/Examples/Monocular/include)
target_compile_options(orbslam3 PRIVATE -Wall)
target_link_libraries(orbslam3 PRIVATE ORB_SLAM3 Boost::program_options rapidcsv::rapidcsv spdlog::spdlog)
