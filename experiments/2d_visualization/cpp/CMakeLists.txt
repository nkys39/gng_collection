cmake_minimum_required(VERSION 3.14)
project(gng_2d_experiments LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include GNG header
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng/cpp)

# Single ring test
add_executable(test_gng_single_ring test_gng_single_ring.cpp)
target_link_libraries(test_gng_single_ring Eigen3::Eigen)

# Tracking test
add_executable(test_gng_tracking test_gng_tracking.cpp)
target_link_libraries(test_gng_tracking Eigen3::Eigen)
