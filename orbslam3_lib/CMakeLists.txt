cmake_minimum_required(VERSION 3.25)

project(ORB_SLAM3 LANGUAGES C CXX)

find_package(Boost CONFIG REQUIRED COMPONENTS serialization)
find_package(DBoW2 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc features2d calib3d imgcodecs video)
find_package(OpenSSL REQUIRED)
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
        source/System.cpp
        source/Tracking.cpp
        source/LocalMapping.cpp
        source/LoopClosing.cpp
        source/keypoint_extractor.cpp
        source/descriptor_matcher.cpp
        source/Converter.cpp
        source/MapPoint.cpp
        source/KeyFrame.cpp
        source/Atlas.cpp
        source/Map.cpp
        source/Optimizer.cpp
        source/Frame.cpp
        source/KeyFrameDatabase.cpp
        source/Sim3Solver.cpp
        source/ImuTypes.cpp
        source/G2oTypes.cpp
        source/CameraModels/Pinhole.cpp
        source/CameraModels/KannalaBrandt8.cpp
        source/OptimizableTypes.cpp
        source/GeometricTools.cpp
        source/TwoViewReconstruction.cpp
        source/Config.cpp
        source/Settings.cpp
        source/DUtils/Random.cpp
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
        opencv_calib3d
        opencv_imgcodecs
        opencv_video
        Eigen3::Eigen
        DBoW2::DBoW2
        Sophus::Sophus
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::types_slam3d
        g2o::types_sim3
        g2o::solver_dense
        g2o::solver_eigen
        Boost::serialization
        OpenSSL::Crypto
)

target_link_libraries(ORB_SLAM3 PRIVATE yaml-cpp::yaml-cpp magic_enum::magic_enum Microsoft.GSL::GSL spdlog::spdlog)
