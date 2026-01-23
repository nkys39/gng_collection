cmake_minimum_required(VERSION 3.14)
project(gng_2d_experiments LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include algorithm headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng_t/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/neural_gas/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng_u/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/som/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gcs/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/hcl/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/lbg/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/growing_grid/cpp)

# Single ring test
add_executable(test_gng_single_ring test_gng_single_ring.cpp)
target_link_libraries(test_gng_single_ring Eigen3::Eigen)

# Tracking test
add_executable(test_gng_tracking test_gng_tracking.cpp)
target_link_libraries(test_gng_tracking Eigen3::Eigen)

# Timing test
add_executable(test_gng_timing test_gng_timing.cpp)
target_link_libraries(test_gng_timing Eigen3::Eigen)

# GNG Triple ring test (50K iterations)
add_executable(test_gng_triple_ring test_gng_triple_ring.cpp)
target_link_libraries(test_gng_triple_ring Eigen3::Eigen)

# GNG-T Triple ring test (50K iterations)
add_executable(test_gngt_triple_ring test_gngt_triple_ring.cpp)
target_link_libraries(test_gngt_triple_ring Eigen3::Eigen)

# Neural Gas Triple ring test
add_executable(test_ng_triple_ring test_ng_triple_ring.cpp)
target_link_libraries(test_ng_triple_ring Eigen3::Eigen)

# GNG-U Triple ring test
add_executable(test_gngu_triple_ring test_gngu_triple_ring.cpp)
target_link_libraries(test_gngu_triple_ring Eigen3::Eigen)

# SOM Triple ring test
add_executable(test_som_triple_ring test_som_triple_ring.cpp)
target_link_libraries(test_som_triple_ring Eigen3::Eigen)

# GCS Triple ring test
add_executable(test_gcs_triple_ring test_gcs_triple_ring.cpp)
target_link_libraries(test_gcs_triple_ring Eigen3::Eigen)

# HCL Triple ring test
add_executable(test_hcl_triple_ring test_hcl_triple_ring.cpp)
target_link_libraries(test_hcl_triple_ring Eigen3::Eigen)

# LBG Triple ring test
add_executable(test_lbg_triple_ring test_lbg_triple_ring.cpp)
target_link_libraries(test_lbg_triple_ring Eigen3::Eigen)

# Growing Grid Triple ring test
add_executable(test_gg_triple_ring test_gg_triple_ring.cpp)
target_link_libraries(test_gg_triple_ring Eigen3::Eigen)

# Kubota variant tests
add_executable(test_gng_kubota_triple_ring test_gng_kubota_triple_ring.cpp)
target_link_libraries(test_gng_kubota_triple_ring Eigen3::Eigen)

add_executable(test_gcs_kubota_triple_ring test_gcs_kubota_triple_ring.cpp)
target_link_libraries(test_gcs_kubota_triple_ring Eigen3::Eigen)

add_executable(test_gngt_kubota_triple_ring test_gngt_kubota_triple_ring.cpp)
target_link_libraries(test_gngt_kubota_triple_ring Eigen3::Eigen)
