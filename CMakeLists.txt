cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_NAME LU_Decomposition)
project(${PROJECT_NAME})

set(PROJ_LIBRARY "${PROJECT_NAME}")
set(PROJ_TESTS   "test_${PROJECT_NAME}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src" gtest)

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(samples)
add_subdirectory(gtest)
add_subdirectory(test)