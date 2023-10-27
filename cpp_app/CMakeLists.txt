cmake_minimum_required(VERSION 3.10)
project(ReadImageProject)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug) # Debug by default

# Find packages
find_package(glfw3 3.3 REQUIRED)  # Find glfw3 package
find_package(OpenGL REQUIRED)  # Find OpenGL package
find_package(OpenCV 4.2 REQUIRED)  # Find OpenCV package
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# Add ImGui (had to be cloned as a submodule)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/dependencies/imgui)

set(IMGUI_SOURCES 
    ${IMGUI_DIR}/imgui.cpp 
    ${IMGUI_DIR}/imgui_demo.cpp 
    ${IMGUI_DIR}/imgui_draw.cpp 
    ${IMGUI_DIR}/imgui_tables.cpp 
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set(IMGUI_INCLUDE_DIRS
    ${IMGUI_DIR} 
    ${IMGUI_DIR}/backends
)
    
# Add mediapipe GPU graph runner
set(MEDIAPIPE_DIR ${CMAKE_SOURCE_DIR}/dependencies/mediapipe)
set(MEDIAPIPE_DESKTOP_DIR ${MEDIAPIPE_DIR}/mediapipe/examples/desktop)

set(MEDIAPIPE_DESKTOP_LIBRARIES 
    ${MEDIAPIPE_DIR}/bazel-bin/mediapipe/examples/desktop/librun_graph_main_gpu.so
)

set(MEDIAPIPE_DESKTOP_INCLUDE_DIRS
    ${MEDIAPIPE_DESKTOP_DIR}
)

# Include packages
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${IMGUI_INCLUDE_DIRS}
    ${MEDIAPIPE_DIR}
    ${MEDIAPIPE_DESKTOP_INCLUDE_DIRS}
)  

add_subdirectory(dependencies/abseil-cpp)

add_executable(main main.cpp ${IMGUI_SOURCES})  # Define executable target
target_link_libraries(main absl::flags absl::log absl::status)

# Link libraries
target_link_libraries(main
    glfw
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLUT_LIBRARIES}
    dl
    ${OpenCV_LIBS}
    ${LIBMP_LIBRARIES}
    ${MEDIAPIPE_DESKTOP_LIBRARIES}
)