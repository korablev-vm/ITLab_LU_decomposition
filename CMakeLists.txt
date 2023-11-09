cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_NAME LU_Decomposition)
project(${PROJECT_NAME})

set(PROJ_LIBRARY      "${PROJECT_NAME}")
set(PROJ_TESTS        "test_${PROJECT_NAME}")
set(PROJ_BENCHMARKS   "benchmarks_${PROJECT_NAME}")

add_subdirectory(extern)

include_directories(include)
add_subdirectory(src)

add_subdirectory(samples)
add_subdirectory(tests)
add_subdirectory(benchmarks)
