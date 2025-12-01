cmake_minimum_required(VERSION 3.24)

project(julia LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /openmp /fp:fast")
    add_compile_definitions(NOMINMAX) 
    add_compile_definitions(GLFW_INCLUDE_NONE) 
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -fopenmp -ffast-math -funroll-loops -fno-math-errno")
endif()

if(WIN32)
    set(NPP_LIBS CUDA::nppif CUDA::nppc)

    include(FetchContent)

    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.2
    )
    FetchContent_MakeAvailable(glm)
else()
    set(NPP_LIBS CUDA::nppif_static CUDA::nppc_static)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --extra-device-vectorization --ptxas-options=-O3 -lineinfo")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 89-real 89-virtual) 

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(julia_lib
    vendor/glad/src/glad.c
    src/avx_kernels.cpp
    src/cuda_kernels.cu
    src/window_2d.cpp
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
    CUDA::cudart_static
    ${NPP_LIBS}
    glm
)

add_executable(animated src/animated2.cpp)
target_link_libraries(animated PRIVATE julia_lib)
set_property(TARGET animated PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_custom_command(TARGET animated POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders
        $<TARGET_FILE_DIR:animated>/shaders
    COMMENT "Copying shaders to binary directory..."
)

if(WIN32)
    add_custom_command(TARGET animated POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:animated>
        $<TARGET_FILE_DIR:animated>
        COMMENT "Copying required DLLS"
        COMMAND_EXPAND_LISTS
    )
endif()
