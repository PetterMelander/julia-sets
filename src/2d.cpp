#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <nppi_statistics_functions.h>

#include "avx_kernels.h"
#include "cuda_kernels.cuh"
#include "window_2d.h"

#ifdef WIN32
#include <windows.h>
extern "C"
{
  __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

NppStreamContext ctx;
CNNModel cnn("medium.trt", "medium.onnx");

void computeJulia(Window2D &window, Npp8u *nppBuffer)
{
  int bufferIndex = window.getNextBufferIndex();
  cudaStream_t stream = window.streams[bufferIndex];
  float *dTargetTex = nullptr;
  float *dPrevTex = nullptr;

  if (!window.paused)
  {
    CUDA_CHECK(cudaGraphicsMapResources(2, window.cudaPboResources, stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void **)&dPrevTex, nullptr, window.cudaPboResources[window.getBufferIndex()]));
  }
  else
    CUDA_CHECK(cudaGraphicsMapResources(1, &window.cudaPboResources[bufferIndex], stream));

  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      (void **)&dTargetTex, nullptr, window.cudaPboResources[bufferIndex]));

  if (window.spSufficient())
    computeJuliaCuda(window.width, window.height, window.c, window.zoomLevel, window.xOffset,
                     window.yOffset, dTargetTex, stream);
  else
  {
    computeJuliaAvx(window.width, window.height, window.c, window.zoomLevel, window.xOffset,
                    window.yOffset, window.hCudaBuffers[bufferIndex]);
    CUDA_CHECK(cudaMemcpyAsync(dTargetTex, window.hCudaBuffers[bufferIndex],
                               window.width * window.height * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  }

  if (!window.paused)
  {
    ctx.hStream = stream;
    NPP_CHECK(nppiAverageRelativeError_32f_C1R_Ctx(
        dTargetTex,
        window.width * sizeof(float),
        dPrevTex,
        window.width * sizeof(float),
        NppiSize{window.width, window.height},
        window.dUpdateRelativeError,
        nppBuffer,
        ctx));
    CUDA_CHECK(cudaMemcpyAsync(
        window.hUpdateRelativeError,
        window.dUpdateRelativeError,
        sizeof(Npp64f),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaGraphicsUnmapResources(2, window.cudaPboResources, stream));
  }
  else
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &window.cudaPboResources[bufferIndex], stream));
}

int main()
{
  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
#endif

  GLFWmonitor *primaryMonitor = glfwGetPrimaryMonitor();
  const GLFWvidmode *mode = glfwGetVideoMode(primaryMonitor);
  glfwWindowHint(GLFW_RED_BITS, mode->redBits);
  glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
  glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
  glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

  int width = mode->width * 0.75;
  int height = mode->height * 0.75;

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
  NppiSize size{width, height};
  size_t nppBufferSize;
  NPP_CHECK(nppiAverageRelativeErrorGetBufferHostSize_32f_C1R_Ctx(size, &nppBufferSize, ctx));
  Npp8u *nppBuffer;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaMalloc(&nppBuffer, nppBufferSize));

  GLFWwindow *windowPtr;
  windowPtr = glfwCreateWindow(width, height, "Julia", NULL, NULL);
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
  Window2D window2d = Window2D(width, height, windowPtr);

  int frameCount = 0;
  auto clock = std::chrono::high_resolution_clock();
  auto start = clock.now();
  while (!glfwWindowShouldClose(window2d.windowPtr))
  {
    window2d.updateState();

    if (window2d.needsRedraw)
    {
      window2d.switchBuffer();
      computeJulia(window2d, nppBuffer);

      window2d.redraw();
      window2d.swap();
    }
    else if (window2d.needsTextureSwitch)
    {
      window2d.switchBuffer();
      window2d.redraw();
      window2d.switchBuffer();
      window2d.swap();
    }
    else
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      --frameCount;
    }

    glfwPollEvents();

    if (++frameCount == 6283)
    {
      auto end = clock.now();
      std::chrono::duration<double> diff = end - start;
      std::cout << "fps: " << 6283 / diff.count() << std::endl;
      frameCount = 0;
      start = clock.now();
    }
  }
  glfwTerminate();
  cudaFree(nppBuffer);

  return 0;
}
