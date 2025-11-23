cmake_minimum_required(VERSION 3.18)

project(julia LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /openmp /fp:fast")
    add_compile_definitions(NOMINMAX) 
    add_compile_definitions(GLFW_INCLUDE_NONE) 
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -fopenmp -ffast-math -funroll-loops -fno-math-errno -fsanitize=address -g")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --extra-device-vectorization --ptxas-options=-O3")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 89) 

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(julia_lib
    vendor/glad/src/glad.c
    src/avx_kernels.cpp
    src/cuda_kernels.cu
    src/gl_utils.cpp
)

target_include_directories(julia_lib PUBLIC
    vendor/glad/include
    include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_property(TARGET julia_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET julia_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(julia_lib PUBLIC
    glfw
    OpenGL::GL 
    CUDA::cudart
)

add_executable(animated src/animated.cpp)
target_link_libraries(animated PRIVATE julia_lib)
set_property(TARGET animated PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_custom_command(TARGET animated POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders
        $<TARGET_FILE_DIR:animated>/shaders
    COMMENT "Copying shaders to binary directory..."
)

add_executable(3d src/3d.cpp)
target_link_libraries(3d PRIVATE julia_lib)
set_property(TARGET 3d PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_custom_command(TARGET 3d POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders
        $<TARGET_FILE_DIR:3d>/shaders
    COMMENT "Copying shaders to binary directory..."
)
