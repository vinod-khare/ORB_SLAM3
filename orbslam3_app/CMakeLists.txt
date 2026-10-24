cmake_minimum_required(VERSION 3.25)

project(orbslam3_app LANGUAGES C CXX)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc features2d)
find_package(hello-imgui CONFIG REQUIRED)
find_package(VTK CONFIG REQUIRED COMPONENTS CommonCore CommonDataModel RenderingCore RenderingOpenGL2)
find_package(rapidcsv CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(orbslam3_input STATIC
    source/folder_reader.cpp
    source/frame_prefetcher.cpp)
target_compile_options(orbslam3_input PRIVATE -Wall)
target_include_directories(orbslam3_input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(orbslam3_input PUBLIC ORB_SLAM3 Threads::Threads PRIVATE rapidcsv::rapidcsv spdlog::spdlog)

add_library(orbslam3_ui STATIC
    source/FrameDrawer.cpp
    source/MapDrawer.cpp
    source/Viewer.cpp
    source/viewer_settings.cpp)
target_compile_options(orbslam3_ui PRIVATE -Wall)
target_include_directories(orbslam3_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(orbslam3_ui PUBLIC ORB_SLAM3 VTK::CommonCore PRIVATE hello-imgui::hello_imgui VTK::CommonDataModel VTK::RenderingCore VTK::RenderingOpenGL2 spdlog::spdlog)
vtk_module_autoinit(TARGETS orbslam3_ui MODULES VTK::RenderingOpenGL2)

hello_imgui_add_app(orbslam3_app
    source/command_line_options.cpp
    source/orbslam3.cpp)

target_compile_options(orbslam3_app PRIVATE -Wall)
target_link_libraries(orbslam3_app PRIVATE orbslam3_input orbslam3_ui Boost::program_options spdlog::spdlog yaml-cpp::yaml-cpp)
