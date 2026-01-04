#pragma once

#include <cmath>
#include <iostream>
#include <memory>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "camera.h"
#include "shader.h"

class Window3D
{
public:
  cudaGraphicsResource *cudaPboResources[2];
  cudaGraphicsResource *cudaVboResources[2];

  Window3D(int width, int height, GLFWwindow *windowPtr);

  inline void updateState() { processMovement(0.01); }

  inline void redraw(bool depthPass = true)
  {
    switchTexture(activeBuffer);
    redrawImage(activeBuffer, depthPass);
  }

  inline void updateView()
  {
    shader->use();
    glm::mat4 transform = camera.GetTransform();
    shader->setMat4("lookAt", transform);
  }

  inline void switchBuffer() { activeBuffer = (activeBuffer + 1) % 2; }

  inline int getBufferIndex() { return activeBuffer; }

  inline int getNextBufferIndex() { return (activeBuffer + 1) % 2; }

  inline void swap() { glfwSwapBuffers(windowPtr); }

  inline bool getNeedsRedraw() { return needsRedraw; }

private:
  GLFWwindow *windowPtr;

  int width;
  int height;

  int activeBuffer = 0;
  bool needsRedraw = true;

  Camera camera{width, height};

  GLuint pboIds[2];
  GLuint heightMap;
  GLuint vaoIds[2];
  GLuint vboIds[2];
  GLuint ebo;
  std::unique_ptr<Shader> shader;

  GLuint depthMapFBO;
  GLuint depthMap;
  unsigned int SHADOW_WIDTH = width * 4;
  unsigned int SHADOW_HEIGHT = height * 4;
  const glm::mat4 lightProjection = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.4f, 1.5f);
  const glm::mat4 lightView = glm::lookAt(glm::vec3(0.4472135955, 0.894427191, 0.0),
                                          glm::vec3(0.0f, 0.0f, 0.0f),
                                          glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::mat4 lightSpaceMatrix = lightProjection * lightView;
  std::unique_ptr<Shader> depthShader;

  void switchTexture(int index);

  void redrawImage(int index, bool depthPass = true);

  void processMovement(float deltaTime);
};
