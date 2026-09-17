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

find_package(OpenCV 4.4)
if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(Pangolin CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(DBoW2 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

if(NOT DBoW2_INCLUDE_DIRS)
        get_filename_component(DBoW2_LIB_DIR ${DBoW2_LIBRARY} DIRECTORY)
        set(DBoW2_INCLUDE_DIRS ${DBoW2_LIB_DIR}/../include)
endif()

set(G2O_LIBS
        g2o_core
        g2o_stuff
        g2o_types_sba
        g2o_types_slam3d
        g2o_types_sim3
        g2o_solver_dense
        g2o_solver_eigen
)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${DBoW2_INCLUDE_DIRS}
)

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
        Thirdparty/DBoW2/DUtils/Random.cpp
        Thirdparty/DBoW2/DUtils/Timestamp.cpp
        include/System.h
        include/Tracking.h
        include/LocalMapping.h
        include/LoopClosing.h
        include/ORBextractor.h
        include/ORBmatcher.h
        include/FrameDrawer.h
        include/Converter.h
        include/MapPoint.h
        include/KeyFrame.h
        include/Atlas.h
        include/Map.h
        include/MapDrawer.h
        include/Optimizer.h
        include/Frame.h
        include/KeyFrameDatabase.h
        include/Sim3Solver.h
        include/Viewer.h
        include/ImuTypes.h
        include/G2oTypes.h
        include/CameraModels/GeometricCamera.h
        include/CameraModels/Pinhole.h
        include/CameraModels/KannalaBrandt8.h
        include/OptimizableTypes.h
        include/MLPnPsolver.h
        include/GeometricTools.h
        include/TwoViewReconstruction.h
        include/SerializationUtils.h
        include/Config.h
        include/Settings.h)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${DBoW2_LIBRARIES}
        Sophus::Sophus
        yaml-cpp::yaml-cpp
        ${G2O_LIBS}
        pango_core
        pango_display
        pango_plot
        pango_video
        -lboost_serialization
        -lcrypto
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
        include_directories(${PROJECT_NAME}
                ${realsense_INCLUDE_DIR}
        )
        target_link_libraries(${PROJECT_NAME}
                ${realsense2_LIBRARY}
        )
endif()


# Build examples

# RGB-D examples

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
        add_executable(rgbd_realsense_D435i
                Examples/RGB-D/rgbd_realsense_D435i.cc)
        target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples

if(realsense2_FOUND)
        add_executable(rgbd_inertial_realsense_D435i
                Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
        target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
        add_executable(stereo_realsense_t265
                Examples/Stereo/stereo_realsense_t265.cc)
        target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

        add_executable(stereo_realsense_D435i
                Examples/Stereo/stereo_realsense_D435i.cc)
        target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

# packages for orbslam3 executable
find_package(rapidcsv CONFIG REQUIRED)

add_executable(orbslam3
        Examples/Monocular/orbslam3.cc
        Examples/Monocular/folder_reader.cc
        include/G2oCompat.h)
target_include_directories(orbslam3 PRIVATE ${PROJECT_SOURCE_DIR}/Examples/Monocular/include)
target_link_libraries(orbslam3 ${PROJECT_NAME} Boost::program_options rapidcsv::rapidcsv spdlog::spdlog)
