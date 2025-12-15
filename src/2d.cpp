#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

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

// TODO: place in other file
void computeJulia_sp_2d(Window2D &window, cudaGraphicsResource *cudaPbo,
                         cudaStream_t stream)
{
  float *dTexBuffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dTexBuffer, nullptr, cudaPbo));

  computeJuliaCuda(window.width, window.height, window.c,
                     window.zoomLevel, window.xOffset, window.yOffset,
                     dTexBuffer, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo, stream));
};

void computeJulia_dp_2d(Window2D &window, float *h_cuda_buffer,
                         cudaGraphicsResource *cudaPbo, cudaStream_t stream)
{
  float *dTexBuffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&dTexBuffer, nullptr, cudaPbo));

  // compute the new julia set into 2d texture
  computeJuliaAvx(window.width, window.height, window.c,
                    window.zoomLevel, window.xOffset, window.yOffset,
                    h_cuda_buffer);
  CUDA_CHECK(cudaMemcpyAsync(dTexBuffer, h_cuda_buffer,
                             window.width * window.height * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo, stream));
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
  #ifndef NDEBUG
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
  #endif
  
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
      int nextBufferIdx = window2d.getNextBufferIndex();
      if (window2d.zoomLevel < 10000)
      {
        computeJulia_sp_2d(window2d, window2d.cudaPboResources[nextBufferIdx],
                            window2d.streams[nextBufferIdx]);
      }
      else
      {
        computeJulia_dp_2d(window2d, window2d.hCudaBuffers[nextBufferIdx],
                            window2d.cudaPboResources[nextBufferIdx],
                            window2d.streams[nextBufferIdx]);
      }

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

  return 0;
}
