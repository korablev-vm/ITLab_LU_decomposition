# This CMakeLists.txt is for your test cases

# Include the GoogleTest library
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Add test cpp file
add_executable(runUnitTests test.cpp)

# Link test executable against gtest & gtest_main
target_link_libraries(runUnitTests gtest gtest_main)