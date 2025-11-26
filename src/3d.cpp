#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <cstddef>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <npp.h>

#include "shader.h"
#include "camera.h"
#include "avx_kernels.h"
#include "gl_utils.h"
#include "cuda_kernels.cuh"

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
  state.width = 1024;
  state.height = 1024;

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
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  // glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetKeyCallback(window, key_callback);

  glViewport(0, 0, state.width, state.height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // init vao to draw on
  int width = 1024;
  int height = 1024;

  std::vector<float> vertices;
  vertices.reserve(2 * width * height);
  std::vector<unsigned int> indices;
  indices.reserve(3 * 2 * (width - 1) * (height - 1));
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      vertices.push_back((float)x / (width - 1) * 2 - 1);
      vertices.push_back((float)y / (height - 1) * 2 - 1);

      if (x < width - 1 && y < height - 1)
      {
        unsigned int i = y * width + x;
        indices.push_back(i);
        indices.push_back(i + 1);
        indices.push_back(i + width + 1);

        indices.push_back(i);
        indices.push_back(i + width + 1);
        indices.push_back(i + width);
      }
    }
  }

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  unsigned int EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  // init shaders
  Shader shader("shaders/shader_3d.vs", "shaders/shader_3d.fs");
  shader.use();

  glEnable(GL_DEPTH_TEST);

  unsigned int heightMapTexture;
  glGenTextures(1, &heightMapTexture);
  glBindTexture(GL_TEXTURE_2D, heightMapTexture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);

  // init pixel buffers
  GLuint pboIds[1];
  int dsize = sizeof(float) * width * height;
  glGenBuffers(1, pboIds);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[0]);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);

  cudaGraphicsResource *cudaPboResources[1];
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResources[0], pboIds[0],
                                          cudaGraphicsMapFlagsWriteDiscard));

  // glm::mat4 model = glm::mat4(1.0f);
  // model = glm::scale(model, glm::vec3(1.0 / sqrt(2.0), 1.0, 1.0 / sqrt(2.0)));
  glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)state.width / (float)state.height, 0.1f, 100.0f);
  // glm::mat4 projection = glm::ortho(-1.0f / sqrtf(2.0f), 1.0f / sqrtf(2.0f), -1.0f / sqrtf(2.0f), 1.0f / sqrtf(2.0f), 0.1f, 100.0f);

  glm::mat4 view = state.camera.GetViewMatrix();

  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  float deltaTime = 0.0f;
  float lastFrame = 0.0f;

  // main render loop
  double length = 0.7885;
  double theta = 0.0;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  while (!glfwWindowShouldClose(window))
  {
    theta += 0.001;
    state.c_re = length * cos(theta);
    state.c_im = length * sin(theta);
    // state.c_re = -0.4;
    // state.c_im = 0.6;

    float *d_buffer = nullptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResources[0], 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_buffer, nullptr, cudaPboResources[0]));
    compute_julia_cuda_smoothed(state, 5, d_buffer, stream);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResources[0], stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    glBindTexture(GL_TEXTURE_2D, heightMapTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, state.width, state.height, GL_RED,
                    GL_FLOAT, 0);

    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    process_movement(window, deltaTime * 0.5);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    shader.use();

    view = state.camera.GetViewMatrix();
    shader.setMat4("lookAt", projection * view);

    glBindTexture(GL_TEXTURE_2D, heightMapTexture);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();
  return 0;
}
