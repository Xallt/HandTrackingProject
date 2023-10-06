cmake_minimum_required(VERSION 3.10)
project(ReadImageProject)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug) # Debug by default

find_package(glfw3 3.3 REQUIRED)  # Find glfw3 package
find_package(OpenGL REQUIRED)  # Find OpenGL package
find_package(OpenCV REQUIRED)  # Find OpenCV package
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
)  

# Include ImGui
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

include_directories(
    ${IMGUI_DIR} 
    ${IMGUI_DIR}/backends
)


# add_executable(main main.cpp)  # Define executable target
add_executable(main main.cpp ${IMGUI_SOURCES})  # Define executable target

target_link_libraries(main
    ${OpenCV_LIBS}
    glfw
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLUT_LIBRARIES}
    dl
)