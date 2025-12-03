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
  GLFWwindow *windowPtr;

  cudaGraphicsResource *cudaPboResources[2];
  cudaGraphicsResource *cudaVboResources[2];

  int width;
  int height;

  int activeBuffer = 0;
  bool needsRedraw = true;
  bool needsTextureSwitch = false;

  Camera camera;

  Window3D(int width, int height);

  void updateState()
  {
    processMovement(0.01);
  }

  void redraw()
  {
    glfwMakeContextCurrent(windowPtr);
    switchTexture(activeBuffer);
    redrawImage(activeBuffer);
  }

  void updateView()
  {
    glm::mat4 view = camera.GetViewMatrix();
    shader->setMat4("lookAt", projection * view);
  }

  void switchBuffer()
  {
    activeBuffer = (activeBuffer + 1) % 2;
  }

  int getBufferIndex() { return activeBuffer; }

  int getNextBufferIndex() { return (activeBuffer + 1) % 2; }

private:
  GLuint pboIds[2];
  GLuint texture;
  GLuint vaoIds[2];
  GLuint vboIds[2];
  GLuint ebo;

  glm::mat4 projection;

  std::unique_ptr<Shader> shader;

  void switchTexture(int index)
  {
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, 0);
  }

  void redrawImage(int index)
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_MULTISAMPLE);

    shader->use();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(vaoIds[index]);
    glDrawElements(GL_TRIANGLES, 3 * 2 * (height - 1) * (width - 1), GL_UNSIGNED_INT, 0);

    glDisable(GL_MULTISAMPLE);

    needsRedraw = false;

    glfwSwapBuffers(windowPtr);
  }

  void processMovement(float deltaTime)
  {
    if (glfwGetKey(windowPtr, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(windowPtr, true);

    if (glfwGetKey(windowPtr, GLFW_KEY_W) == GLFW_PRESS)
    {
      camera.ProcessKeyboard(FORWARD, deltaTime);
      shader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
    if (glfwGetKey(windowPtr, GLFW_KEY_S) == GLFW_PRESS)
    {
      camera.ProcessKeyboard(BACKWARD, deltaTime);
      shader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
    if (glfwGetKey(windowPtr, GLFW_KEY_A) == GLFW_PRESS)
    {
      camera.ProcessKeyboard(LEFT, deltaTime);
      shader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
    if (glfwGetKey(windowPtr, GLFW_KEY_D) == GLFW_PRESS)
    {
      camera.ProcessKeyboard(RIGHT, deltaTime);
      shader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
  }
};
