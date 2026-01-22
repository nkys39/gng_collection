cmake_minimum_required(VERSION 3.14)
project(gng_collection VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect algorithm implementations
file(GLOB ALGO_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../algorithms/*/cpp")
foreach(ALGO_DIR ${ALGO_DIRS})
    if(EXISTS "${ALGO_DIR}/CMakeLists.txt")
        get_filename_component(ALGO_NAME ${ALGO_DIR} DIRECTORY)
        get_filename_component(ALGO_NAME ${ALGO_NAME} NAME)
        message(STATUS "Found algorithm: ${ALGO_NAME}")
        add_subdirectory(${ALGO_DIR} ${CMAKE_BINARY_DIR}/algorithms/${ALGO_NAME})
    endif()
endforeach()

# Testing
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
        message(STATUS "Google Test found, building tests")
    else()
        message(STATUS "Google Test not found, tests will not be built")
    endif()
endif()

# Installation
install(DIRECTORY include/gng DESTINATION include)
