#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "callbacks.h"
#include "cuda_kernels.cuh"
#include "shader.h"
#include "window_3d.h"

Window3D::Window3D(int width, int height) : width(width), height(height)
{

  // create window
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  windowPtr = glfwCreateWindow(width, height, "Julia", NULL, NULL);
  if (windowPtr == NULL)
  {
    std::cout << "Failed to create 2D window" << std::endl;
    glfwTerminate();
  }
  glfwMakeContextCurrent(windowPtr);

  // allocate buffers, texture, etc.
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
      vertices.push_back(0.0f);
      vertices.push_back(0.0f);

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

  constexpr size_t aoNumSamples = 16;
  std::uniform_real_distribution<float> randomAngles(0.0, 2 * glm::pi<float>());
  std::uniform_real_distribution<float> randomRadii(0.0, 1.0);
  std::default_random_engine generator;
  std::array<glm::vec2, aoNumSamples> aoKernel;
  for (unsigned int i = 0; i < aoNumSamples; ++i)
  {
    float a = randomAngles(generator);
    glm::vec2 sample(cosf(a), sinf(a));
    sample = sample * sqrtf(randomRadii(generator));
    float scale = (float)i / (float)aoNumSamples;
    scale = 0.1f + scale * scale * 0.9f;
    aoKernel[i] = sample * scale;
  }

  glGenVertexArrays(2, vaoIds);
  glGenBuffers(2, pboIds);
  glGenBuffers(2, vboIds);
  glGenBuffers(1, &ebo);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
               indices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  for (int i = 0; i < 2; ++i)
  {
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

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
  }

  glGenTextures(1, &heightMap);
  glBindTexture(GL_TEXTURE_2D, heightMap);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);

  // set up resources for shadows
  glGenFramebuffers(1, &depthMapFBO);
  glGenTextures(1, &depthMap);
  glBindTexture(GL_TEXTURE_2D, depthMap);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0,
    GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

  glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  depthShader = std::make_unique<Shader>("shaders/shader_3d_shadow.vs", "shaders/shader_3d_shadow.fs");
  depthShader->use();
  depthShader->setMat4("lightSpaceMatrix", lightSpaceMatrix);
  depthShader->setInt("heightMap", 1);

  // set window parameters and callbacks
  glfwSwapInterval(0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  // glEnable(GL_FRAMEBUFFER_SRGB);
  glFrontFace(GL_CW);
  glfwSetWindowUserPointer(windowPtr, this);

  glViewport(0, 0, width, height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  glfwSetFramebufferSizeCallback(windowPtr, framebufferSizeCallback);

  // init shader
  shader = std::make_unique<Shader>("shaders/shader_3d.vs", "shaders/shader_3d.fs");
  shader->use();
  shader->setInt("heightMap", 1);
  shader->setFloat("xstep", 2.0 / (float)(width - 1));
  shader->setFloat("ystep", 2.0 / (float)(height - 1));
  shader->setVec3("viewPos", camera.front);
  shader->setMat4("lightSpaceMatrix", lightSpaceMatrix);
  shader->setInt("shadowMap", 2);
  for (unsigned int i = 0; i < aoNumSamples; ++i)
  {
    shader->setVec2("aoSamples[" + std::to_string(i) + "]", aoKernel[i]);
  }
}
