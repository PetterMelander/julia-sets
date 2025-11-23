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
extern "C"
{
  __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

int main()
{
  // set initial state
  ProgramState state;
  state.width = 512;
  state.height = 512;

  // width must be multiple of 8 for avx kernel to work
  state.width = (state.width + 7) / 8 * 8;

  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(state.width, state.height, "Julia", NULL, NULL);
  if (window == NULL)
  {
    std::cout << "Failed to create window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // set user input callbacks
  glfwSetWindowUserPointer(window, &state);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetKeyCallback(window, key_callback);

  glViewport(0, 0, state.width, state.height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // init cuda buffers & streams
  // buffers are used for double precision color mapping
  int raw_dsize = state.width * state.height * sizeof(float);
  cudaStream_t streams[2];
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));

  float *h_cuda_buffers[2];
  cudaMallocHost(&h_cuda_buffers[0], raw_dsize);
  cudaMallocHost(&h_cuda_buffers[1], raw_dsize);

  float *d_cuda_buffers[2];
  cudaMalloc(&d_cuda_buffers[0], raw_dsize);
  cudaMalloc(&d_cuda_buffers[1], raw_dsize);

  // init pixel buffers
  // used directly for single precision
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

  // init texture and vao to draw on
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, state.width, state.height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, 0);

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  float vertices[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f,
                      -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f};
  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  unsigned int indices[] = {0, 1, 3, 1, 2, 3};
  unsigned int EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // init shaders
  Shader shader("shaders/shader.vs", "shaders/shader.fs");
  shader.use();
  shader.setInt("texture1", 0);

  // main render loop
  int bufIdx = 0;
  bool needs_texture_switch = false;
  glfwSwapInterval(0);
  double R = sqrt(3.0);
  double r = 2.2;
  double d = 0.3;
  while (!glfwWindowShouldClose(window))
  {
    update_pan(state, window);
    update_theta(state, window);

    if (state.needs_redraw)
    {
      // compute next fractal and display previously computed fractal
      state.c_re = (R - r) * cos(state.theta) + d * cos((R - r) * state.theta / r);
      state.c_im = (R - r) * sin(state.theta) - d * sin((R - r) * state.theta / r);


      bufIdx = (bufIdx + 1) % 2;
      int nextIdx = (bufIdx + 1) % 2;

      if (state.zoomLevel < 10000)
      {
        compute_julia_sp(state, cudaPboResources[nextIdx], streams[nextIdx]);
      }
      else
      {
        compute_julia_dp(state, h_cuda_buffers[nextIdx], d_cuda_buffers[nextIdx],
                         cudaPboResources[nextIdx], streams[nextIdx]);
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

      // make sure previous fractal is finished before rendering
      CUDA_CHECK(cudaStreamSynchronize(streams[bufIdx]));
      switch_texture(state, bufIdx, texture, pboIds);
      redraw_image(window, shader, texture, VAO);

      state.needs_redraw = false;
      needs_texture_switch = true;
    }
    else if (needs_texture_switch)
    {
      // if we just paused, render last computed fractal, then stop
      int nextIdx = (bufIdx + 1) % 2;

      // make sure previous fractal is finished before rendering
      CUDA_CHECK(cudaStreamSynchronize(streams[bufIdx]));
      switch_texture(state, nextIdx, texture, pboIds);
      redraw_image(window, shader, texture, VAO);

      needs_texture_switch = false;
    }

    glfwPollEvents();
  }
  glfwTerminate();

  cudaFree(d_cuda_buffers[0]);
  cudaFree(d_cuda_buffers[1]);
  cudaFreeHost(h_cuda_buffers[0]);
  cudaFreeHost(h_cuda_buffers[1]);

  return 0;
}
