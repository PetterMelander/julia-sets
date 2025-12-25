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
#include "mlp.h"
#include "window_2d.h"

#include "cnn.h"
#include "mlp_constants.h"
#include "xgb.h"

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

  float result = 0;
  
  // enqueue cnn pred
  cnn.enqueue(window.c, window.zoomLevel, window.xOffset, window.yOffset);

  // xgb pred
  std::vector<Entry> entries(5);
  entries[0].fvalue = ((float)window.c.real() - INPUT_MEANS[0]) / INPUT_STDS[0];
  entries[1].fvalue = ((float)window.c.imag() - INPUT_MEANS[1]) / INPUT_STDS[1];
  entries[2].fvalue = ((float)window.xOffset - INPUT_MEANS[2]) / INPUT_STDS[2];
  entries[3].fvalue = ((float)window.yOffset - INPUT_MEANS[3]) / INPUT_STDS[3];
  entries[4].fvalue = ((float)log(window.zoomLevel * 1.331 * window.width / 224.0 + 1.0) - INPUT_MEANS[4]) / INPUT_STDS[4];
  predict(entries.data(), 0, &result);

  // mlp pred
  double windowParams[] = {window.c.real(), window.c.imag(), window.xOffset, window.yOffset, window.zoomLevel}; 
  result += 1.0f / (1.0f + expf(-mlpPredict(windowParams)));
  
  // get cnn pred
  result += 2.0f * 1.0f / (1.0f + expf(-cnn.getPred()));
  
  // weight ensemble 1 - 1 - 2
  result /= 4.0f;
  if (result >= 0.5f)
    computeJuliaCuda(window.width, window.height, window.c, window.zoomLevel, window.xOffset,
                     window.yOffset, dTargetTex, stream);
  else
  {
    std::cout << "insufficient precision" << std::endl;
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

  int width = 224;
  int height = 224;

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
