cmake_minimum_required(VERSION 3.14)
project(gsrm_experiments LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3 - try system install first, fall back to eigency
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    # Fallback: use eigency's Eigen headers
    execute_process(
        COMMAND python3 -c "import eigency; print(eigency.get_includes()[-1])"
        OUTPUT_VARIABLE EIGENCY_EIGEN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EIGENCY_RESULT
    )
    if(EIGENCY_RESULT EQUAL 0)
        message(STATUS "Using Eigen from eigency: ${EIGENCY_EIGEN_PATH}")
        include_directories(${EIGENCY_EIGEN_PATH})
        set(EIGEN_FOUND TRUE)
    else()
        message(FATAL_ERROR "Neither Eigen3 package nor eigency found")
    endif()
else()
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# Include algorithm headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../algorithms/gsrm/cpp)

# Include helper headers from 3d_pointcloud
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3d_pointcloud/cpp)

# GSRM sphere test
add_executable(test_gsrm_sphere test_gsrm_sphere.cpp)
if(Eigen3_FOUND)
    target_link_libraries(test_gsrm_sphere Eigen3::Eigen)
endif()

# GSRM torus test
add_executable(test_gsrm_torus test_gsrm_torus.cpp)
if(Eigen3_FOUND)
    target_link_libraries(test_gsrm_torus Eigen3::Eigen)
endif()
