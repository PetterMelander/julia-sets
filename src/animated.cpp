#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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
  state.width = 1536;
  state.height = 1536;

  // width must be multiple of 8 for avx kernel to work
  state.width = (state.width + 7) / 8 * 8;

  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window_2d =
      glfwCreateWindow(state.width, state.height, "Julia", NULL, NULL);
  if (window_2d == NULL)
  {
    std::cout << "Failed to create window_2d" << std::endl;
    glfwTerminate();
    return -1;
  }
  GLFWwindow *window_3d =
      glfwCreateWindow(state.width, state.height, "Julia", NULL, window_2d);
  if (window_3d == NULL)
  {
    std::cout << "Failed to create window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window_2d);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

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

  // init pixel buffers for 2d
  // used directly for single precision
  GLuint colorPboIds[2];
  int dsize = sizeof(float) * state.width * state.height;
  glGenBuffers(2, colorPboIds);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, colorPboIds[0]);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, colorPboIds[1]);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);

  cudaGraphicsResource *colorCudaPboResources[2];
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&colorCudaPboResources[0], colorPboIds[0],
                                          cudaGraphicsMapFlagsWriteDiscard));
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&colorCudaPboResources[1], colorPboIds[1],
                                          cudaGraphicsMapFlagsWriteDiscard));

  // init pixel buffers for 3d
  // used directly for single precision
  GLuint smoothPboIds[2];
  dsize = sizeof(float) * state.width * state.height;
  glGenBuffers(2, smoothPboIds);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, smoothPboIds[0]);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, smoothPboIds[1]);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);

  cudaGraphicsResource *smoothCudaPboResources[2];
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&smoothCudaPboResources[0], smoothPboIds[0],
                                          cudaGraphicsMapFlagsWriteDiscard));
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&smoothCudaPboResources[1], smoothPboIds[1],
                                          cudaGraphicsMapFlagsWriteDiscard));

  // init texture
  unsigned int texture_2d;
  glGenTextures(1, &texture_2d);
  glBindTexture(GL_TEXTURE_2D, texture_2d);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, state.width, state.height, 0, GL_RED,
               GL_FLOAT, 0);

  /*
  ==============================
        2D WINDOW SETUP
  ==============================
  */
  glfwSwapInterval(0);
  glfwSetWindowUserPointer(window_2d, &state);

  // set user input callbacks
  glfwSetScrollCallback(window_2d, scroll_callback);
  glfwSetMouseButtonCallback(window_2d, mouse_button_callback);
  glfwSetKeyCallback(window_2d, key_callback);

  glViewport(0, 0, state.width, state.height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glfwSetFramebufferSizeCallback(window_2d, framebuffer_size_callback);

  unsigned int VAO_2d;
  glGenVertexArrays(1, &VAO_2d);
  glBindVertexArray(VAO_2d);

  float vertices_2d[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f,
                         -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f};
  unsigned int VBO_2d;
  glGenBuffers(1, &VBO_2d);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_2d);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_2d), vertices_2d, GL_STATIC_DRAW);

  unsigned int indices_2d[] = {0, 1, 3, 1, 2, 3};
  unsigned int EBO_2d;
  glGenBuffers(1, &EBO_2d);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_2d);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_2d), indices_2d, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // init shaders
  Shader shader_2d("shaders/shader.vs", "shaders/shader.fs");
  shader_2d.use();
  shader_2d.setInt("texture1", 0);

  /*
  ==============================
        3D WINDOW SETUP
  ==============================
  */

  glfwMakeContextCurrent(window_3d);
  glPointSize(1.5);

  glfwSetWindowUserPointer(window_3d, &state);

  glViewport(0, 0, state.width, state.height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glfwSetFramebufferSizeCallback(window_3d, framebuffer_size_callback);

  std::vector<float> vertices_3d;
  vertices_3d.reserve(2 * state.width * state.height);
  std::vector<unsigned int> indices_3d;
  indices_3d.reserve(3 * 2 * (state.width - 1) * (state.height - 1));
  for (int y = 0; y < state.height; ++y)
  {
    for (int x = 0; x < state.width; ++x)
    {
      vertices_3d.push_back((float)x / (state.width - 1) * 2 - 1);
      vertices_3d.push_back((float)y / (state.height - 1) * 2 - 1);

      if (x < state.width - 1 && y < state.height - 1)
      {
        unsigned int i = y * state.width + x;
        indices_3d.push_back(i);
        indices_3d.push_back(i + 1);
        indices_3d.push_back(i + state.width + 1);

        indices_3d.push_back(i);
        indices_3d.push_back(i + state.width + 1);
        indices_3d.push_back(i + state.width);
      }
    }
  }

  unsigned int VAO_3d;
  glGenVertexArrays(1, &VAO_3d);
  glBindVertexArray(VAO_3d);

  unsigned int VBO_3d;
  glGenBuffers(1, &VBO_3d);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_3d);
  glBufferData(GL_ARRAY_BUFFER, vertices_3d.size() * sizeof(float), vertices_3d.data(), GL_STATIC_DRAW);

  unsigned int EBO_3d;
  glGenBuffers(1, &EBO_3d);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_3d);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_3d.size() * sizeof(unsigned int), indices_3d.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  unsigned int texture_3d;
  glGenTextures(1, &texture_3d);
  glBindTexture(GL_TEXTURE_2D, texture_3d);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, state.width, state.height, 0, GL_RED, GL_FLOAT, NULL);

  // init shaders
  Shader shader_3d("shaders/shader_3d.vs", "shaders/shader_3d.fs");
  shader_3d.use();
  shader_3d.setInt("texture2", 1);

  glEnable(GL_DEPTH_TEST);

  // main render loop
  int bufIdx = 0;
  bool needs_texture_switch = false;
  double R = sqrt(3.0);
  double r = 2.2;
  double d = 0.3;

  glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)state.width / (float)state.height, 0.1f, 100.0f);
  glm::mat4 view = state.camera.GetViewMatrix();

  while (!glfwWindowShouldClose(window_2d))
  {
    update_pan(state, window_2d);
    process_movement(window_2d, 0.005);
    update_theta(state, window_2d);

    if (state.needs_redraw)
    {
      // compute next fractal and display previously computed fractal
      state.c_re = (R - r) * cos(state.theta) + d * cos((R - r) * state.theta / r);
      state.c_im = (R - r) * sin(state.theta) - d * sin((R - r) * state.theta / r);

      bufIdx = (bufIdx + 1) % 2;
      int nextIdx = (bufIdx + 1) % 2;

      if (state.zoomLevel < 10000)
      {
        compute_julia_sp(
            state,
            colorCudaPboResources[nextIdx],
            smoothCudaPboResources[nextIdx],
            streams[nextIdx]);
      }
      else
      {
        compute_julia_dp(state, h_cuda_buffers[nextIdx], d_cuda_buffers[nextIdx],
                         colorCudaPboResources[nextIdx], streams[nextIdx]);
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

      // make sure previous fractal is finished before rendering
      CUDA_CHECK(cudaStreamSynchronize(streams[bufIdx]));

      glfwMakeContextCurrent(window_2d);
      switch_texture(state, bufIdx, texture_2d, colorPboIds);
      redraw_image(window_2d, shader_2d, texture_2d, VAO_2d);

      glfwMakeContextCurrent(window_3d);
      view = state.camera.GetViewMatrix();
      shader_3d.setMat4("lookAt", projection * view);
      switch_texture_3d(state, nextIdx, texture_3d, smoothPboIds);
      redraw_image_3d(window_3d, shader_3d, texture_3d, VAO_3d);

      state.needs_redraw = false;
      needs_texture_switch = true;
    }
    else if (needs_texture_switch)
    {
      // if we just paused, render last computed fractal, then stop
      int nextIdx = (bufIdx + 1) % 2;

      // make sure previous fractal is finished before rendering
      CUDA_CHECK(cudaStreamSynchronize(streams[bufIdx]));

      glfwMakeContextCurrent(window_2d);
      switch_texture(state, nextIdx, texture_2d, colorPboIds);
      redraw_image(window_2d, shader_2d, texture_2d, VAO_2d);

      glfwMakeContextCurrent(window_3d);
      view = state.camera.GetViewMatrix();
      shader_3d.setMat4("lookAt", projection * view);
      switch_texture_3d(state, nextIdx, texture_3d, smoothPboIds);
      redraw_image_3d(window_3d, shader_3d, texture_3d, VAO_3d);

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
