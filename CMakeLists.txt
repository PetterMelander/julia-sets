cmake_minimum_required(VERSION 3.18)

project(julia)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -fopenmp -ffast-math")

add_library(julia_lib
    vendor/glad/src/glad.c
    src/stb_image.cpp
    src/julia_avx.cpp
    src/gl_utils.cpp
)

target_include_directories(julia_lib PUBLIC
    vendor/glad/include
    vendor/stb/include
    include
)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(julia_lib PRIVATE glfw GL)

add_executable(animated src/animated.cpp)
target_link_libraries(animated PRIVATE julia_lib)
set_property(TARGET animated PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

