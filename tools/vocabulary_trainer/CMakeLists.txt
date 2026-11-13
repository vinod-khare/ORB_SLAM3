find_package(DBow3 CONFIG QUIET)
if(NOT DBow3_FOUND)
    message(STATUS "Skipping vocabulary_trainer: DBoW3 is not available")
    return()
endif()

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(magic_enum CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_executable(vocabulary_trainer main.cpp)
target_compile_options(vocabulary_trainer PRIVATE -Wall -Wextra)
target_link_libraries(vocabulary_trainer PRIVATE ORB_SLAM3 DBow3::DBow3 Boost::program_options magic_enum::magic_enum spdlog::spdlog)