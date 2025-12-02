
#include <cmath>
#include <iostream>
#include <memory>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "callbacks.h"
#include "cuda_kernels.cuh"
#include "shader.h"
#include "window_2d.h"

void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id,
                            GLenum severity, GLsizei length,
                            const char *message, const void *userParam) {
  // Ignore non-significant error/warning codes
  if (id == 131169 || id == 131185 || id == 131218 || id == 131204)
    return;

  std::cout << "---------------" << std::endl;
  std::cout << "Debug message (" << id << "): " << message << std::endl;

  switch (source) {
  case GL_DEBUG_SOURCE_API:
    std::cout << "Source: API";
    break;
  case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
    std::cout << "Source: Window System";
    break;
  case GL_DEBUG_SOURCE_SHADER_COMPILER:
    std::cout << "Source: Shader Compiler";
    break;
  case GL_DEBUG_SOURCE_THIRD_PARTY:
    std::cout << "Source: Third Party";
    break;
  case GL_DEBUG_SOURCE_APPLICATION:
    std::cout << "Source: Application";
    break;
  case GL_DEBUG_SOURCE_OTHER:
    std::cout << "Source: Other";
    break;
  }
  std::cout << std::endl;

  switch (type) {
  case GL_DEBUG_TYPE_ERROR:
    std::cout << "Type: Error";
    break;
  case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
    std::cout << "Type: Deprecated Behaviour";
    break;
  case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
    std::cout << "Type: Undefined Behaviour";
    break;
  case GL_DEBUG_TYPE_PORTABILITY:
    std::cout << "Type: Portability";
    break;
  case GL_DEBUG_TYPE_PERFORMANCE:
    std::cout << "Type: Performance";
    break;
  case GL_DEBUG_TYPE_MARKER:
    std::cout << "Type: Marker";
    break;
  case GL_DEBUG_TYPE_PUSH_GROUP:
    std::cout << "Type: Push Group";
    break;
  case GL_DEBUG_TYPE_POP_GROUP:
    std::cout << "Type: Pop Group";
    break;
  case GL_DEBUG_TYPE_OTHER:
    std::cout << "Type: Other";
    break;
  }
  std::cout << std::endl;

  std::cout << "---------------" << std::endl;
}

Window2D::Window2D(int width, int height) : width(width), height(height) {

  // create window
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 0);

#ifdef NDEBUG
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
#endif

  window_ptr = glfwCreateWindow(width, height, "Julia", NULL, NULL);
  if (window_ptr == NULL) {
    std::cout << "Failed to create 2D window" << std::endl;
    glfwTerminate();
  }
  glfwMakeContextCurrent(window_ptr);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
  }

#ifdef NDEBUG
  int flags;
  glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
  if (flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); // Makes sure errors are displayed at
                                           // the moment they happen
    glDebugMessageCallback(glDebugOutput, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr,
                          GL_TRUE);
  }
#endif

  // allocate buffers, texture, etc.
  int dsize = width * height * sizeof(float);
  glGenBuffers(2, pboIds);
  for (int i = 0; i < 2; ++i) {

    cudaMallocHost(&h_cuda_buffers[i], dsize);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[i]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResources[i], pboIds[i],
                                            cudaGraphicsMapFlagsNone));
  }

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT,
               0);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  float vertices[] = {1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
                      -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f,  0.0f, 1.0f};
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  unsigned int indices[] = {0, 1, 3, 1, 2, 3};
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // set window parameters and callbacks
  glfwSwapInterval(0);
  glfwSetWindowUserPointer(window_ptr, this);

  glViewport(0, 0, width, height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  glfwSetScrollCallback(window_ptr, scroll_callback);
  glfwSetMouseButtonCallback(window_ptr, mouse_button_callback);
  glfwSetKeyCallback(window_ptr, key_callback);
  glfwSetFramebufferSizeCallback(window_ptr, framebuffer_size_callback);

  // init shader
  shader = std::make_unique<Shader>("shaders/shader.vs", "shaders/shader.fs");
  shader->use();
  shader->setInt("texture1", 0);
}

Window2D::~Window2D() {
  cudaFreeHost(h_cuda_buffers[0]);
  cudaFreeHost(h_cuda_buffers[1]);
}
