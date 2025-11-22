#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <chrono>
#include "shader.h"
#include "avx_kernels.h"
#include "cuda_kernels.cuh"
#include "gl_utils.h"

#ifdef WIN32
#include <windows.h>
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

int main() {

  ProgramState state;
  state.width = 1024;
  state.height = 1024;
  state.c_re = 0.35;
  state.c_im = 0.35;

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(state.width, state.height, "Julia", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  glfwSetWindowUserPointer(window, &state);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetKeyCallback(window, key_callback);

  glViewport(0, 0, state.width, state.height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  int raw_dsize = state.width * state.height * sizeof(float);
  cudaStream_t streams[3];
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));
  CUDA_CHECK(cudaStreamCreate(&streams[2]));

  float *h_cuda_buffers[2];
  cudaMallocHost(&h_cuda_buffers[0], raw_dsize);
  cudaMallocHost(&h_cuda_buffers[1], raw_dsize);

  float *d_cuda_buffers[2];
  cudaMalloc(&d_cuda_buffers[0], raw_dsize);
  cudaMalloc(&d_cuda_buffers[1], raw_dsize);

  GLuint pboIds[2];
  int dsize = sizeof(unsigned char) * state.width * state.height * 3;
  glGenBuffers(2, pboIds);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[0]);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[1]);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);

  cudaGraphicsResource *cudaPboResources[2];
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResources[0], pboIds[0],
                                          cudaGraphicsMapFlagsWriteDiscard));
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResources[1], pboIds[1],
                                          cudaGraphicsMapFlagsWriteDiscard));

  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, state.width, state.height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, 0);

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  float vertices[] = {1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
                      -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f,  0.0f, 1.0f};
  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  unsigned int indices[] = {0, 1, 3, 1, 2, 3};
  unsigned int EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  Shader shader("shaders/shader.vs", "shaders/shader.fs");
  shader.use();
  shader.setInt("texture1", 0);

  double length = 0.7885;
  int buffer_index = 0;
  bool needs_texture_switch = false;
  auto clock = std::chrono::high_resolution_clock();
  auto start = clock.now();
  int frames = 0;
  glfwSwapInterval(0);
  while (!glfwWindowShouldClose(window)) {
    update_pan(state, window);

    process_fractal_update(state, window);

    if (state.needs_redraw) {
      state.c_re = length * cos(state.theta);
      state.c_im = length * sin(state.theta);

      buffer_index = (buffer_index + 1) % 2;
      int nextIndex = (buffer_index + 1) % 2;

      if (state.zoomLevel < 10000) {
        unsigned char *d_buffer = nullptr;
        CUDA_CHECK(
            cudaGraphicsMapResources(1, &cudaPboResources[nextIndex], 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            (void **)&d_buffer, nullptr, cudaPboResources[nextIndex]));
        compute_julia_cuda(state, d_buffer);
        CUDA_CHECK(
            cudaGraphicsUnmapResources(1, &cudaPboResources[nextIndex], 0));
      } else {
        compute_julia_avx(state, h_cuda_buffers[nextIndex]);
        cudaMemcpyAsync(d_cuda_buffers[nextIndex], h_cuda_buffers[nextIndex],
                        raw_dsize, cudaMemcpyHostToDevice, streams[nextIndex]);

        unsigned char *d_buffer = nullptr;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResources[nextIndex],
                                            streams[2]));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            (void **)&d_buffer, nullptr, cudaPboResources[nextIndex]));

        map_colors_cuda(d_buffer, d_cuda_buffers[nextIndex],
                        state.width * state.height, streams[nextIndex]);
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResources[nextIndex],
                                              streams[nextIndex]));
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

      state.needs_redraw = false;
      needs_texture_switch = true;

      CUDA_CHECK(cudaStreamSynchronize(streams[buffer_index]));
      switch_texture(state, buffer_index, texture, pboIds);

      redraw_image(window, shader, texture, VAO);
      ++frames;
    } else if (needs_texture_switch) {
      int nextIndex = (buffer_index + 1) % 2;
      switch_texture(state, nextIndex, texture, pboIds);
      needs_texture_switch = false;

      redraw_image(window, shader, texture, VAO);
      ++frames;
    }
    if (frames == 6283) {
      auto end = clock.now();
      const std::chrono::duration<double> diff = end - start;
      std::cout << "fps: " << 6283 / diff.count() << std::endl;
      frames = 0;
      start = clock.now();
    }

    glfwPollEvents();
  }
  glfwTerminate();
  return 0;
}
