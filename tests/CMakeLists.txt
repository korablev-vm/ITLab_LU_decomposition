set(target ${PROJ_TESTS})

file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp")

enable_testing()

add_executable(${target} ${srcs} ${hdrs})

target_link_libraries(${target} GTest::gtest_main ${PROJ_LIBRARY})

include(GoogleTest)
gtest_discover_tests(${target})