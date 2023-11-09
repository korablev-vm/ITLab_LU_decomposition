set(target ${PROJ_BENCHMARKS})

file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp")

add_executable(${target} ${srcs} ${hdrs})

target_link_libraries(${target} benchmark::benchmark ${PROJ_LIBRARY})