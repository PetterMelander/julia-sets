cmake_minimum_required(VERSION 3.24)

if(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

project(julia LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /openmp /fp:fast /openmp:llvm /openmp:experimental")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /openmp /fp:fast /openmp:llvm /openmp:experimental")
    add_compile_definitions(NOMINMAX) 
    add_compile_definitions(GLFW_INCLUDE_NONE) 
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -fopenmp -ffast-math -funroll-loops -fno-math-errno")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -march=native -fopenmp -ffast-math -funroll-loops -fno-math-errno")
endif()

if(WIN32)
    set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT root directory")
    
    find_path(TRT_INCLUDE_DIR NAMES NvInfer.h
        PATHS ${TENSORRT_ROOT} PATH_SUFFIXES include
    )
    find_library(TRT_LIB NAMES nvinfer_10 nvinfer PATHS ${TENSORRT_ROOT} PATH_SUFFIXES lib)
    find_library(TRT_ONNX_LIB NAMES nvonnxparser_10 nvonnxparser PATHS ${TENSORRT_ROOT} PATH_SUFFIXES lib)

    set(TRT_LIBRARIES ${TRT_LIB} ${TRT_ONNX_LIB})
    
    if(NOT TRT_LIB)
        message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT to your TensorRT install folder.")
    endif()
else()
    set(TRT_LIBRARIES nvinfer nvonnxparser)
endif()

if(WIN32)
    set(NPP_LIBS CUDA::nppif CUDA::nppist CUDA::nppc)

    include(FetchContent)

    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.2
    )
    FetchContent_MakeAvailable(glm)
else()
    set(NPP_LIBS CUDA::nppif_static CUDA::nppist_static CUDA::nppc_static)
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
    src/xgb.c
    src/avx_kernels.cpp
    src/cuda_kernels.cu
    src/window_2d.cpp
    src/window_3d.cpp
    src/labeling.cpp
    src/cnn.cpp
    src/imsave.cpp
)

target_include_directories(julia_lib PUBLIC
    include
    vendor/glad/include
    vendor/stb/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)

set_property(TARGET julia_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET julia_lib PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_link_libraries(julia_lib PUBLIC
    glfw
    OpenGL::GL
    ${TRT_LIBRARIES}
    CUDA::cudart_static
    ${NPP_LIBS}
    glm
)

macro(setup_executable name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE julia_lib)
    set_property(TARGET ${name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

    # 1. Copy Shaders
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders
            $<TARGET_FILE_DIR:${name}>/shaders
        COMMENT "Copying shaders..."
    )
    
    # 2. Copy ONNX Model
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/precision_labeling/cnn.onnx
            $<TARGET_FILE_DIR:${name}>/cnn.onnx
        COMMENT "Copying onnx model..."
    )

    if(WIN32)
        # 3. Copy vcpkg/System DLLs (Generic)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${name}>
            $<TARGET_FILE_DIR:${name}>
            COMMAND_EXPAND_LISTS
            COMMENT "Copying runtime DLLs..."
        )

        # 4. Copy NPP DLLs
        if(CUDAToolkit_FOUND)
            file(GLOB NPP_DLLS 
                "${CUDAToolkit_BIN_DIR}/x64/nppif64_*.dll"
                "${CUDAToolkit_BIN_DIR}/x64/nppist64_*.dll"
                "${CUDAToolkit_BIN_DIR}/x64/nppc64_*.dll"
            )
            
            if(NPP_DLLS)
                add_custom_command(TARGET ${name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${NPP_DLLS}
                    $<TARGET_FILE_DIR:${name}>
                    COMMENT "Copying NPP DLLs..."
                )
            endif()
        endif()

        # 5. Copy TensorRT DLLs
        if(TENSORRT_ROOT)
            add_custom_command(TARGET ${name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${TENSORRT_ROOT}/bin/nvinfer_10.dll"
                "${TENSORRT_ROOT}/bin/nvonnxparser_10.dll"
                "${TENSORRT_ROOT}/bin/nvinfer_builder_resource_sm89_10.dll"
                $<TARGET_FILE_DIR:${name}>
                COMMENT "Copying TensorRT DLLs..."
            )
        endif()
    endif()

    if(WIN32 AND CUDAToolkit_FOUND)
        set_property(TARGET ${name} PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=${CUDAToolkit_BIN_DIR};%PATH%")
    endif()
endmacro()

setup_executable(2d)
setup_executable(3d)