cmake_minimum_required(VERSION 3.14)
project(LU_Decomposition)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(extern/googletest)
include_directories(extern/googletest/googletest/include)
add_subdirectory(extern/benchmark)
include_directories(include)
add_subdirectory(tests)
add_subdirectory(benchmarks)

add_executable(LU_Decomposition src/Matrix.cpp benchmarks/benchmarks.cpp src/Matrix.cpp tests/test.cpp include/Matrix.h)
