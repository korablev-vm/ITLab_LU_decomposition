cmake_minimum_required(VERSION 2.8)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

include_directories(base gtest)

# BUILD
add_subdirectory(base)
add_subdirectory(base_test)
add_subdirectory(gtest)

# Добавление поддиректории Google Benchmark
add_subdirectory(externals/benchmark)

# Установка флага для использования Google Benchmark без установки тестов
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)

# Подключение директорий include Google Benchmark
include_directories(externals/benchmark/include)
