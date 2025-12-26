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

  void switchTexture(int index)
  {
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, heightMap);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, 0);
  }

  void redrawImage(int index, bool depthPass = true)
  {

    if (depthPass)
    {
      depthShader->use();
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, heightMap);

      glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
      glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
      glClear(GL_DEPTH_BUFFER_BIT);
      glBindVertexArray(vaoIds[index]);
      glDrawElements(GL_TRIANGLES, 3 * 2 * (height - 1) * (width - 1), GL_UNSIGNED_INT, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // main pass
    glViewport(width, 0, width, height);
    glEnable(GL_MULTISAMPLE);

    shader->use();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, heightMap);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, depthMap);

    glBindVertexArray(vaoIds[index]);
    glDrawElements(GL_TRIANGLES, 3 * 2 * (height - 1) * (width - 1), GL_UNSIGNED_INT, 0);

    glDisable(GL_MULTISAMPLE);

    needsRedraw = false;
  }

  void processMovement(float deltaTime)
  {
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
