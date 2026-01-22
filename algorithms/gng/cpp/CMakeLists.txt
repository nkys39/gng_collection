cmake_minimum_required(VERSION 3.14)

# Library
add_library(gng_algorithm INTERFACE)
target_include_directories(gng_algorithm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gng_algorithm INTERFACE Eigen3::Eigen)

# Tests
if(BUILD_TESTS AND GTest_FOUND)
    add_executable(test_gng tests/test_gng.cpp)
    target_link_libraries(test_gng gng_algorithm GTest::gtest_main)
    add_test(NAME test_gng COMMAND test_gng)
endif()
