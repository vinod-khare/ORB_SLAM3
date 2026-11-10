cmake_minimum_required(VERSION 3.25)

project(ORB_SLAM3 LANGUAGES C CXX)

find_package(DBoW2 CONFIG REQUIRED)
find_package(DBow3 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc features2d calib3d imgcodecs video xfeatures2d)
find_package(Sophus CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

if(NOT DBoW2_INCLUDE_DIRS)
        find_path(DBoW2_INCLUDE_DIRS NAMES DBoW2/DBoW2.h)
endif()

if(NOT TARGET DBoW2::DBoW2)
        add_library(DBoW2::DBoW2 UNKNOWN IMPORTED)
        set_target_properties(DBoW2::DBoW2 PROPERTIES
                IMPORTED_LOCATION "${DBoW2_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${DBoW2_INCLUDE_DIRS}"
        )
endif()

add_library(ORB_SLAM3 SHARED
        source/place_vocabulary.cpp
        source/dbow2_vocabulary.cpp
        source/dbow3_vocabulary.cpp
        source/system.cpp
        source/tracking.cpp
        source/local_mapping.cpp
        source/loop_closing.cpp
        source/keypoint_extractor.cpp
        source/descriptor_matcher.cpp
        source/converter.cpp
        source/map_point.cpp
        source/key_frame.cpp
        source/atlas.cpp
        source/map.cpp
        source/optimizer.cpp
        source/frame.cpp
        source/key_frame_database.cpp
        source/sim3_solver.cpp
        source/imu_types.cpp
        source/g2o_types.cpp
        source/CameraModels/pinhole.cpp
        source/CameraModels/kannala_brandt8.cpp
        source/optimizable_types.cpp
        source/geometric_tools.cpp
        source/two_view_reconstruction.cpp
        source/settings.cpp
        source/DUtils/random.cpp
        source/DUtils/Timestamp.cpp)

target_compile_definitions(ORB_SLAM3 PUBLIC EIGEN_MAX_ALIGN_BYTES=16)
target_compile_options(ORB_SLAM3 PRIVATE -Wall)

target_include_directories(ORB_SLAM3 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CameraModels
)

target_link_libraries(ORB_SLAM3 PUBLIC
        opencv_core
        opencv_imgproc
        opencv_features2d
        opencv_xfeatures2d
        opencv_calib3d
        opencv_imgcodecs
        opencv_video
        Eigen3::Eigen
        DBoW2::DBoW2
        DBow3::DBow3
        Sophus::Sophus
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::types_slam3d
        g2o::types_sim3
        g2o::solver_dense
        g2o::solver_eigen
)

target_link_libraries(ORB_SLAM3 PRIVATE yaml-cpp::yaml-cpp magic_enum::magic_enum Microsoft.GSL::GSL spdlog::spdlog)
