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
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

// TODO: place in other file
void compute_julia_sp_2d(Window2D &window, cudaGraphicsResource *cudaPbo,
                         cudaStream_t stream) {
  float *d_color_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_color_buffer,
                                                  nullptr, cudaPbo));

  compute_julia_cuda(window.width, window.height, window.c_re, window.c_im,
                     window.zoomLevel, window.x_offset, window.y_offset,
                     d_color_buffer, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo, stream));
};

void compute_julia_dp_2d(Window2D &window, float *h_cuda_buffer,
                         cudaGraphicsResource *cudaPbo, cudaStream_t stream) {
  float *d_color_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_color_buffer,
                                                  nullptr, cudaPbo));

  // compute the new julia set into 2d texture
  compute_julia_avx(window.width, window.height, window.c_re, window.c_im,
                    window.zoomLevel, window.x_offset, window.y_offset,
                    h_cuda_buffer);
  CUDA_CHECK(cudaMemcpyAsync(d_color_buffer, h_cuda_buffer,
                             window.width * window.height * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo, stream));
}

int main() {
  int width = 2048;
  int height = 2048;

  // width must be multiple of 8 for avx kernel to work
  width = (width + 7) / 8 * 8;

  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  Window2D window_2d = Window2D(width, height);

  cudaStream_t streams[2];
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));

  while (!glfwWindowShouldClose(window_2d.window_ptr)) {
    window_2d.update_state();

    if (window_2d.needs_redraw) {
      int nextIdx = window_2d.getNextBufferIndex();
      if (window_2d.zoomLevel < 10000) {
        compute_julia_sp_2d(window_2d, window_2d.cudaPboResources[nextIdx],
                            streams[nextIdx]);
      } else {
        compute_julia_dp_2d(window_2d, window_2d.h_cuda_buffers[nextIdx],
                            window_2d.cudaPboResources[nextIdx],
                            streams[nextIdx]);
      }

      CUDA_CHECK(cudaStreamSynchronize(streams[window_2d.active_buffer]));
      window_2d.redraw();
    } else if (window_2d.needs_texture_switch) {
      int nextIdx = window_2d.getNextBufferIndex();
      CUDA_CHECK(cudaStreamSynchronize(streams[nextIdx]));
      window_2d.redraw();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    glfwPollEvents();
  }
  glfwTerminate();

  return 0;
}
