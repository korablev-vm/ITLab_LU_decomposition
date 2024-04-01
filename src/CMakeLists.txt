set(target ${PROJ_LIBRARY})

# Укажите путь к заголовочным файлам
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Соберите список заголовочных файлов из ../include
file(GLOB hdrs "${CMAKE_SOURCE_DIR}/../include/*.h*")

# Соберите список исходных файлов
file(GLOB srcs "*.cpp")

# Создайте библиотеку из заголовочных и исходных файлов
add_library(${target} STATIC ${srcs} ${hdrs})

# Свяжите библиотеку с другими библиотеками, если это необходимо
target_link_libraries(${target} ${LIBRARY_DEPS})
