# This CMakeLists.txt is for your benchmark cases

# Add benchmark executable
add_executable(runBenchmarks benchmarks.cpp)

# Link benchmark executable against the benchmark library
target_link_libraries(runBenchmarks benchmark::benchmark)
