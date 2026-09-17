cmake_minimum_required(VERSION 3.11)

if(POLICY CMP0074)
        cmake_policy(SET CMP0074 NEW)
endif()

project(ORB_SLAM3)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=16)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(DBoW2 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc features2d calib3d imgcodecs highgui)
find_package(Pangolin CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
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

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Atlas.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Viewer.cc
        src/ImuTypes.cc
        src/G2oTypes.cc
        src/CameraModels/Pinhole.cpp
        src/CameraModels/KannalaBrandt8.cpp
        src/OptimizableTypes.cpp
        src/MLPnPsolver.cpp
        src/GeometricTools.cc
        src/TwoViewReconstruction.cc
        src/Config.cc
        src/Settings.cc
        src/DUtils/Random.cpp
        src/DUtils/Timestamp.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
)

target_link_libraries(${PROJECT_NAME}
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
        yaml-cpp::yaml-cpp
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::types_slam3d
        g2o::types_sim3
        g2o::solver_dense
        g2o::solver_eigen
        -lboost_serialization
        -lcrypto
)


# packages for orbslam3 executable
find_package(rapidcsv CONFIG REQUIRED)

add_executable(orbslam3
        Examples/Monocular/orbslam3.cc
        Examples/Monocular/folder_reader.cc)
target_include_directories(orbslam3 PRIVATE ${PROJECT_SOURCE_DIR}/Examples/Monocular/include)
target_link_libraries(orbslam3 ${PROJECT_NAME} Boost::program_options rapidcsv::rapidcsv spdlog::spdlog)
