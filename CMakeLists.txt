cmake_minimum_required(VERSION 3.10)
project(ReadImageProject)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)  # Find OpenCV package
include_directories(${OpenCV_INCLUDE_DIRS})  # Include OpenCV directories
add_executable(main main.cpp)  # Define executable target
target_link_libraries(main ${OpenCV_LIBS})  # Link OpenCV libraries to the target
