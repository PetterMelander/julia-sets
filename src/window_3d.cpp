
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
#include "window_3d.h"

Window3D::Window3D(int width, int height) : width(width), height(height) {

  // create window
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 16);

  window_ptr = glfwCreateWindow(width, height, "Julia", NULL, NULL);
  if (window_ptr == NULL) {
    std::cout << "Failed to create 2D window" << std::endl;
    glfwTerminate();
  }
  glfwMakeContextCurrent(window_ptr);

  // if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
  //   std::cout << "Failed to initialize GLAD" << std::endl;
  // }

  // allocate buffers, texture, etc.
  std::vector<float> vertices;
  vertices.reserve(2 * width * height);
  std::vector<unsigned int> indices;
  indices.reserve(3 * 2 * (width - 1) * (height - 1));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      vertices.push_back((float)x / (width - 1) * 2 - 1);
      vertices.push_back((float)y / (height - 1) * 2 - 1);
      vertices.push_back(0.0f);
      vertices.push_back(0.0f);

      if (x < width - 1 && y < height - 1) {
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

  int dsize = width * height * sizeof(float);
  glGenVertexArrays(2, vaoIds);
  glGenBuffers(2, pboIds);
  glGenBuffers(2, vboIds);
  glGenBuffers(1, &ebo);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
               indices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  for (int i = 0; i < 2; ++i) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[i]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_DYNAMIC_DRAW);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResources[i], pboIds[i],
                                            cudaGraphicsMapFlagsWriteDiscard));

    glBindVertexArray(vaoIds[i]);

    glBindBuffer(GL_ARRAY_BUFFER, vboIds[i]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_DYNAMIC_DRAW);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVboResources[i], vboIds[i],
                                            cudaGraphicsMapFlagsWriteDiscard));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
  }

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT,
               0);

  // set window parameters and callbacks
  glfwSwapInterval(1);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glFrontFace(GL_CW);
  glfwSetWindowUserPointer(window_ptr, this);

  glViewport(0, 0, width, height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  glfwSetFramebufferSizeCallback(window_ptr, framebuffer_size_callback);

  // init shader
  shader =
      std::make_unique<Shader>("shaders/shader_3d.vs", "shaders/shader_3d.fs");
  shader->use();
  shader->setInt("texture2", 1);
  shader->setFloat("xstep", 2.0 / (float)(width - 1));
  shader->setFloat("ystep", 2.0 / (float)(height - 1));
  shader->setVec3("viewPos", camera.Front);

  projection = glm::perspective(glm::radians(45.0f),
                                (float)width / (float)height, 0.1f, 100.0f);
}
