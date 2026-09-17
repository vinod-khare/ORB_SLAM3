cmake_minimum_required(VERSION 3.11)

project(ORB_SLAM3 LANGUAGES C CXX)

find_package(Boost CONFIG REQUIRED COMPONENTS serialization)
find_package(DBoW2 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc features2d calib3d imgcodecs highgui)
find_package(OpenSSL REQUIRED)
find_package(Pangolin CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

if(NOT DBoW2_INCLUDE_DIRS)
        get_filename_component(DBoW2_LIB_DIR ${DBoW2_LIBRARY} DIRECTORY)
        set(DBoW2_INCLUDE_DIRS ${DBoW2_LIB_DIR}/../include)
endif()
if(NOT TARGET DBoW2::DBoW2)
        add_library(DBoW2::DBoW2 UNKNOWN IMPORTED)
        set_target_properties(DBoW2::DBoW2 PROPERTIES
                IMPORTED_LOCATION "${DBoW2_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${DBoW2_INCLUDE_DIRS}"
        )
endif()

add_library(ORB_SLAM3 SHARED
        source/System.cc
        source/Tracking.cc
        source/LocalMapping.cc
        source/LoopClosing.cc
        source/ORBextractor.cc
        source/ORBmatcher.cc
        source/FrameDrawer.cc
        source/Converter.cc
        source/MapPoint.cc
        source/KeyFrame.cc
        source/Atlas.cc
        source/Map.cc
        source/MapDrawer.cc
        source/Optimizer.cc
        source/Frame.cc
        source/KeyFrameDatabase.cc
        source/Sim3Solver.cc
        source/Viewer.cc
        source/ImuTypes.cc
        source/G2oTypes.cc
        source/CameraModels/Pinhole.cpp
        source/CameraModels/KannalaBrandt8.cpp
        source/OptimizableTypes.cpp
        source/MLPnPsolver.cpp
        source/GeometricTools.cc
        source/TwoViewReconstruction.cc
        source/Config.cc
        source/Settings.cc
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
        opencv_highgui
        Eigen3::Eigen
        pango_display
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

target_link_libraries(ORB_SLAM3 PRIVATE yaml-cpp::yaml-cpp)
