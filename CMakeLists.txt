cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PTCPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Torch REQUIRED)

include_directories(include)

add_executable(example1 src/example-1.cc)
target_link_libraries(example1 "${TORCH_LIBRARIES}")
add_executable(example2 src/example-2.cc include/PTNET.h)
target_link_libraries(example2 "${TORCH_LIBRARIES}")
add_executable(example3 src/example-3.cc include/PTNET.h)
target_link_libraries(example3 "${TORCH_LIBRARIES}")