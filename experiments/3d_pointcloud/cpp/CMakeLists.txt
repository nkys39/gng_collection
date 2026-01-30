cmake_minimum_required(VERSION 3.14)
project(gng_3d_experiments LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include algorithm headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng_u/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng_u2/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/ais_gng/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gng_dt/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/ais_gng_dt/cpp)

# GNG floor wall test
add_executable(test_gng_floor_wall test_gng_floor_wall.cpp)
target_link_libraries(test_gng_floor_wall Eigen3::Eigen)

# GNG-U floor wall test
add_executable(test_gngu_floor_wall test_gngu_floor_wall.cpp)
target_link_libraries(test_gngu_floor_wall Eigen3::Eigen)

# GNG-DT floor wall test
add_executable(test_gngdt_floor_wall test_gngdt_floor_wall.cpp)
target_link_libraries(test_gngdt_floor_wall Eigen3::Eigen)

# GNG-DT Robot floor wall test
add_executable(test_gngdt_robot_floor_wall test_gngdt_robot_floor_wall.cpp)
target_link_libraries(test_gngdt_robot_floor_wall Eigen3::Eigen)

# AiS-GNG floor wall test
add_executable(test_aisgng_floor_wall test_aisgng_floor_wall.cpp)
target_link_libraries(test_aisgng_floor_wall Eigen3::Eigen)

# AiS-GNG-AM floor wall test
add_executable(test_aisgng_am_floor_wall test_aisgng_am_floor_wall.cpp)
target_link_libraries(test_aisgng_am_floor_wall Eigen3::Eigen)

# AiS-GNG-RO-MAN floor wall test
add_executable(test_aisgng_roman_floor_wall test_aisgng_roman_floor_wall.cpp)
target_link_libraries(test_aisgng_roman_floor_wall Eigen3::Eigen)

# AiS-GNG-DT floor wall test
add_executable(test_aisgngdt_floor_wall test_aisgngdt_floor_wall.cpp)
target_link_libraries(test_aisgngdt_floor_wall Eigen3::Eigen)
