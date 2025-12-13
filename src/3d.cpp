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

// TODO: place in other file
void computeJulia_sp_3d(Window2D &window, cudaGraphicsResource *cudaPbo2d,
                        cudaGraphicsResource *cudaPbo3d, cudaGraphicsResource *cudaVbo3d,
                        float *dImgMax, float *dImgMin, Npp8u *dNppistBuffer, cudaStream_t stream)
{
  float *dTexBuffer2d = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo2d, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dTexBuffer2d, nullptr, cudaPbo2d));

  float *dTexBuffer3d = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo3d, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dTexBuffer3d, nullptr, cudaPbo3d));

  float *dVboBuffer3d = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVbo3d, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dVboBuffer3d, nullptr, cudaVbo3d));

  computeJuliaCuda(window.width, window.height, window.c,
                   window.zoomLevel, window.xOffset, window.yOffset,
                   dTexBuffer2d, stream);

  NppiSize size = {window.width, window.height};
  ctx.hStream = stream;
  NPP_CHECK(nppiFilterGaussBorder_32f_C1R_Ctx(
      dTexBuffer2d, sizeof(float) * window.width, size, NppiPoint{0, 0},
      dTexBuffer3d, sizeof(float) * window.width, size, NPP_MASK_SIZE_9_X_9,
      NPP_BORDER_REPLICATE, ctx));
  NPP_CHECK(nppiMinMax_32f_C1R_Ctx(dTexBuffer3d, sizeof(float) * window.width, size, dImgMin, dImgMax, dNppistBuffer, ctx));
  rescaleImage(window.width, window.height, dImgMin, dImgMax, dTexBuffer3d, stream);
  computeNormalsCuda(window.width, window.height, dTexBuffer3d, dVboBuffer3d, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo2d, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo3d, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVbo3d, stream));
};

void computeJulia_dp_3d(Window2D &window, float *hCudaBuffer,
                        cudaGraphicsResource *cudaPbo2d, cudaGraphicsResource *cudaPbo3d,
                        cudaGraphicsResource *cudaVbo3d, float *dImgMax, float *dImgMin,
                        Npp8u *dNppistBuffer, cudaStream_t stream)
{
  float *dTexBuffer2d = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo2d, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dTexBuffer2d, nullptr, cudaPbo2d));

  float *dTexBuffer3d = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo3d, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dTexBuffer3d, nullptr, cudaPbo3d));

  float *dVboBuffer3d = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVbo3d, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dVboBuffer3d, nullptr, cudaVbo3d));

  // compute the new julia set into 2d texture
  computeJuliaAvx(window.width, window.height, window.c,
                  window.zoomLevel, window.xOffset, window.yOffset,
                  hCudaBuffer);
  CUDA_CHECK(cudaMemcpyAsync(dTexBuffer2d, hCudaBuffer,
                             window.width * window.height * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  NppiSize size = {window.width, window.height};
  NppStreamContext ctx;
  ctx.hStream = stream;
  NPP_CHECK(nppiFilterGaussBorder_32f_C1R_Ctx(
      dTexBuffer2d, sizeof(float) * window.width, size, NppiPoint{0, 0},
      dTexBuffer3d, sizeof(float) * window.width, size, NPP_MASK_SIZE_9_X_9,
      NPP_BORDER_REPLICATE, ctx));
  NPP_CHECK(nppiMinMax_32f_C1R_Ctx(dTexBuffer3d, sizeof(float) * window.width, size, dImgMin, dImgMax, dNppistBuffer, ctx));
  rescaleImage(window.width, window.height, dImgMin, dImgMax, dTexBuffer3d, stream);
  // float tmp;
  // cudaMemcpy(&tmp, dImgMin, sizeof(float), cudaMemcpyDeviceToHost);
  // std::cout << tmp << std::endl;
  computeNormalsCuda(window.width, window.height, dTexBuffer3d, dVboBuffer3d, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo2d, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo3d, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVbo3d, stream));
}

int main()
{
  int width = 2048;
  int height = 2048;

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
  NppiSize size = {width, height};
  NPP_CHECK(nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(size, &minMaxBufferSize, ctx));
  Npp8u *dNppistBuffer;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMalloc(&dNppistBuffer, minMaxBufferSize));

  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  Window2D window2d = Window2D(width, height);
  Window3D window3d = Window3D(width, height);

  int frameCount = 0;
  auto clock = std::chrono::high_resolution_clock();
  auto start = clock.now();
  while (!glfwWindowShouldClose(window2d.windowPtr))
  {
    window2d.updateState();
    window3d.updateState();

    if (window2d.needsRedraw)
    {
      window2d.switchBuffer();
      window3d.switchBuffer();

      int nextBufferIdx = window2d.getNextBufferIndex();
      if (window2d.zoomLevel < 10000)
      {
        computeJulia_sp_3d(window2d,
                           window2d.cudaPboResources[nextBufferIdx],
                           window3d.cudaPboResources[nextBufferIdx],
                           window3d.cudaVboResources[nextBufferIdx],
                           dImgMax,
                           dImgMin,
                           dNppistBuffer,
                           window2d.streams[nextBufferIdx]);
      }
      else
      {
        computeJulia_dp_3d(window2d,
                           window2d.hCudaBuffers[nextBufferIdx],
                           window2d.cudaPboResources[nextBufferIdx],
                           window3d.cudaPboResources[nextBufferIdx],
                           window3d.cudaVboResources[nextBufferIdx],
                           dImgMax,
                           dImgMin,
                           dNppistBuffer,
                           window2d.streams[nextBufferIdx]);
      }

      window2d.redraw();
      window3d.redraw();
      window3d.updateView();
      window3d.swap();
      window2d.swap();
    }
    else if (window2d.needsTextureSwitch)
    {
      window2d.switchBuffer();
      window2d.redraw();
      window2d.switchBuffer();

      window3d.switchBuffer();
      window3d.updateView();
      window3d.redraw();
      window3d.switchBuffer();
      window3d.swap();
      window2d.swap();
    }
    else if (window3d.needsRedraw)
    {
      window3d.switchBuffer();
      window3d.updateView();
      window3d.redraw();
      window3d.swap();
      window3d.switchBuffer();
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

  CUDA_CHECK(cudaFree(dImgMax));
  CUDA_CHECK(cudaFree(dImgMin));
  CUDA_CHECK(cudaFree(dNppistBuffer));

  return 0;
}
