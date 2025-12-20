#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <nppcore.h>
#include <nppi_filtering_functions.h>
#include <nppi_statistics_functions.h>

#include "avx_kernels.h"
#include "cuda_kernels.cuh"
#include "window_2d.h"
#include "window_3d.h"

#ifdef WIN32
#include <windows.h>
extern "C"
{
  __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

NppStreamContext ctx;

void computeJulia(Window2D &window2D, Window3D &window3D,
                  float *dImgMin, float *dImgMax, Npp8u *nppBuffer)
{
  int bufferIndex = window2D.getNextBufferIndex();
  cudaStream_t stream = window2D.streams[bufferIndex];

  // Map cuda buffers
  cudaGraphicsResource *cudaResources[4];
  cudaResources[0] = window3D.cudaPboResources[bufferIndex];
  cudaResources[1] = window3D.cudaVboResources[bufferIndex];
  cudaResources[2] = window2D.cudaPboResources[bufferIndex];
  float *dTargetTex2D = nullptr;
  float *dPrevTex2D = nullptr;
  float *dTex3D = nullptr;
  float *dVbo3D = nullptr;

  // If unpaused, map previous julia set for rate-of-change calculations
  if (!window2D.paused)
  {
    cudaResources[3] = window2D.cudaPboResources[window2D.getBufferIndex()];
    CUDA_CHECK(cudaGraphicsMapResources(4, cudaResources, stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void **)&dPrevTex2D, nullptr, window2D.cudaPboResources[window2D.getBufferIndex()]));
  }
  else
    CUDA_CHECK(cudaGraphicsMapResources(3, cudaResources, stream));

  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      (void **)&dTargetTex2D, nullptr, window2D.cudaPboResources[bufferIndex]));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      (void **)&dTex3D, nullptr, window3D.cudaPboResources[bufferIndex]));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      (void **)&dVbo3D, nullptr, window3D.cudaVboResources[bufferIndex]));

  // Compute new julia set
  if (window2D.zoomLevel < 10000.0)
    computeJuliaCuda(window2D.width, window2D.height, window2D.c, window2D.zoomLevel,
                     window2D.xOffset, window2D.yOffset, dTargetTex2D, stream);
  else
  {
    computeJuliaAvx(window2D.width, window2D.height, window2D.c, window2D.zoomLevel,
                    window2D.xOffset, window2D.yOffset, window2D.hCudaBuffers[bufferIndex]);
    CUDA_CHECK(cudaMemcpyAsync(dTargetTex2D, window2D.hCudaBuffers[bufferIndex],
                               window2D.width * window2D.height * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  }

  // If unpaused, calculate rate of change. Unmap previous julia set immediately for next iteration
  NppiSize size{window2D.width, window2D.height};
  ctx.hStream = stream;
  if (!window2D.paused)
  {
    NPP_CHECK(nppiAverageRelativeError_32f_C1R_Ctx(
        dTargetTex2D,
        window2D.width * sizeof(float),
        dPrevTex2D,
        window2D.width * sizeof(float),
        size,
        window2D.dUpdateRelativeError,
        nppBuffer,
        ctx));
    CUDA_CHECK(cudaMemcpyAsync(
        window2D.hUpdateRelativeError,
        window2D.dUpdateRelativeError,
        sizeof(Npp64f),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResources[3], stream));
  }

  // Gauss filter, rescale height, and compute normals for lighting
  NPP_CHECK(nppiFilterGaussBorder_32f_C1R_Ctx(
      dTargetTex2D, sizeof(float) * window2D.width, size, NppiPoint{0, 0},
      dTex3D, sizeof(float) * window2D.width, size, NPP_MASK_SIZE_5_X_5,
      NPP_BORDER_REPLICATE, ctx));
  size = {window2D.width / 2, window2D.height / 2};
  NPP_CHECK(nppiMinMax_32f_C1R_Ctx(
      dTex3D + window2D.height / 4 * window2D.width + window2D.width / 4,
      sizeof(float) * window2D.width, size, dImgMin, dImgMax, nppBuffer, ctx));
  rescaleImage(window2D.width, window2D.height, dImgMin, dImgMax, dTex3D, stream);
  computeNormalsCuda(window2D.width, window2D.height, dTex3D, dVbo3D, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(3, cudaResources, stream));
}

int main()
{
  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 16);
  #ifndef NDEBUG
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
  #endif
  
  GLFWmonitor *primaryMonitor = glfwGetPrimaryMonitor();
  const GLFWvidmode *mode = glfwGetVideoMode(primaryMonitor);
  glfwWindowHint(GLFW_RED_BITS, mode->redBits);
  glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
  glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
  glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

  int width = mode->width / 2;
  int height = mode->height;

  // width must be multiple of 8 for avx kernel to work
  width = (width + 7) / 8 * 8;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  int deviceId;
  CUDA_CHECK(cudaGetDevice(&deviceId));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
  ctx.nCudaDeviceId = deviceId;
  ctx.nMultiProcessorCount = props.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = props.sharedMemPerBlock;
  ctx.hStream = stream;
  float *dImgMax, *dImgMin;
  CUDA_CHECK(cudaMalloc(&dImgMax, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dImgMin, sizeof(float)));
  size_t minMaxBufferSize;
  NppiSize size{width / 2, height / 2};
  NPP_CHECK(nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(size, &minMaxBufferSize, ctx));
  size = NppiSize{width, height};
  size_t errorBufferSize;
  NPP_CHECK(nppiAverageRelativeErrorGetBufferHostSize_32f_C1R_Ctx(size, &errorBufferSize, ctx));
  Npp8u *nppBuffer;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaMalloc(&nppBuffer, std::max(minMaxBufferSize, errorBufferSize)));


  GLFWwindow *windowPtr;
  windowPtr = glfwCreateWindow(width * 2, height, "Julia", primaryMonitor, NULL);
  if (windowPtr == NULL)
  {
    std::cout << "Failed to create window" << std::endl;
    glfwTerminate();
  }
  glfwMakeContextCurrent(windowPtr);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
  }
  Window2D window2D = Window2D(width, height, windowPtr);
  Window3D window3D = Window3D(width, height, windowPtr);

  int frameCount = 0;
  auto clock = std::chrono::high_resolution_clock();
  auto start = clock.now();
  while (!glfwWindowShouldClose(window2D.windowPtr))
  {
    ++frameCount;
    window2D.updateState();
    window3D.updateState();

    if (window2D.needsRedraw)
    {
      window2D.switchBuffer();
      window3D.switchBuffer();
      computeJulia(window2D, window3D, dImgMin, dImgMax, nppBuffer);

      window2D.redraw();
      window3D.redraw();
      window3D.updateView();

      window2D.swap();
    }
    else if (window2D.needsTextureSwitch)
    {
      window2D.switchBuffer();
      window2D.redraw();
      window2D.switchBuffer();

      window3D.switchBuffer();
      window3D.updateView();
      window3D.redraw();
      window3D.switchBuffer();

      window2D.swap();
    }
    else if (window3D.needsRedraw)
    {
      window3D.switchBuffer();
      window3D.updateView();
      window2D.redraw(false);
      window3D.redraw(false);
      window3D.switchBuffer();

      window3D.swap();
    }
    else
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      --frameCount;
    }

    glfwPollEvents();

    if (window2D.theta >= 2.0 * glm::pi<double>())
    {
      auto end = clock.now();
      std::chrono::duration<double> diff = end - start;
      std::cout << "fps: " << frameCount / diff.count() << std::endl;
      frameCount = 0;
      // window2D.theta = 0.0;
      start = clock.now();
    }
  }
  glfwTerminate();

  CUDA_CHECK(cudaFree(dImgMax));
  CUDA_CHECK(cudaFree(dImgMin));
  CUDA_CHECK(cudaFree(nppBuffer));

  return 0;
}
