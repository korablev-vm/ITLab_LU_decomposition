set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST OFF CACHE BOOL "" FORCE)


add_subdirectory(googletest EXCLUDE_FROM_ALL)
add_subdirectory(benchmark EXCLUDE_FROM_ALL)

set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL FALSE)
set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL FALSE)
set_target_properties(benchmark PROPERTIES EXCLUDE_FROM_ALL FALSE)
set_target_properties(benchmark_main PROPERTIES EXCLUDE_FROM_ALL FALSE)