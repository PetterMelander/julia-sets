#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <nppi_filtering_functions.h>

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

// TODO: place in other file
void computeJulia_sp_3d(Window2D &window, cudaGraphicsResource *cudaPbo2d, cudaGraphicsResource *cudaPbo3d,
                         cudaGraphicsResource *cudaVbo3d, cudaStream_t stream)
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
  NppStreamContext ctx;
  ctx.hStream = stream;
  nppiFilterGaussBorder_32f_C1R_Ctx(
      dTexBuffer2d, sizeof(float) * window.width, size, NppiPoint{0, 0},
      dTexBuffer3d, sizeof(float) * window.width, size, NPP_MASK_SIZE_9_X_9,
      NPP_BORDER_REPLICATE, ctx);
  computeNormalsCuda(window.width, window.height, dTexBuffer3d, dVboBuffer3d, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo2d, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo3d, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVbo3d, stream));
};

void computeJulia_dp_3d(Window2D &window, float *h_cuda_buffer,
                         cudaGraphicsResource *cudaPbo2d, cudaGraphicsResource *cudaPbo3d,
                         cudaGraphicsResource *cudaVbo3d, cudaStream_t stream)
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
                    h_cuda_buffer);
  CUDA_CHECK(cudaMemcpyAsync(dTexBuffer2d, h_cuda_buffer,
                             window.width * window.height * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  NppiSize size = {window.width, window.height};
  NppStreamContext ctx;
  ctx.hStream = stream;
  nppiFilterGaussBorder_32f_C1R_Ctx(
      dTexBuffer2d, sizeof(float) * window.width, size, NppiPoint{0, 0},
      dTexBuffer3d, sizeof(float) * window.width, size, NPP_MASK_SIZE_9_X_9,
      NPP_BORDER_REPLICATE, ctx);
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

  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  Window2D window2d = Window2D(width, height);
  Window3D window3d = Window3D(width, height);

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
                           window2d.streams[nextBufferIdx]);
      }
      else
      {
        computeJulia_dp_3d(window2d,
                           window2d.hCudaBuffers[nextBufferIdx],
                           window2d.cudaPboResources[nextBufferIdx],
                           window3d.cudaPboResources[nextBufferIdx],
                           window3d.cudaVboResources[nextBufferIdx],
                           window2d.streams[nextBufferIdx]);
      }

      window2d.redraw();
      window3d.redraw();
      window3d.updateView();
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
    }
    else if (window3d.needsRedraw)
    {
      window3d.updateView();
      window3d.redraw();
    }
    else
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    glfwPollEvents();
  }
  glfwTerminate();

  return 0;
}
