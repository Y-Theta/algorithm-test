cmake_minimum_required(VERSION 3.10)
# set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
# include(CMakeForceCompiler)
project(algorithm)

set(CMAKE_CXX_STANDARD 17)

# Add all source files
file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/main.cpp"
    "${CMAKE_SOURCE_DIR}/cppResolve/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cpp"
)

add_executable(algorithm ${SOURCES})

# Include directories if needed
target_include_directories(algorithm PRIVATE
    "${CMAKE_SOURCE_DIR}/cppResolve"
    "${CMAKE_SOURCE_DIR}/test"
)