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

  Camera camera{width, height};

  Window3D(int width, int height);

  void updateState()
  {
    processMovement(0.01);
  }

  void redraw()
  {
    if (glfwGetCurrentContext() != windowPtr)
      glfwMakeContextCurrent(windowPtr);
    switchTexture(activeBuffer);
    redrawImage(activeBuffer);
  }

  void updateView()
  {
    if (glfwGetCurrentContext() != windowPtr)
      glfwMakeContextCurrent(windowPtr);
    glm::mat4 transform = camera.GetTransform();
    mainShader->use();
    mainShader->setMat4("lookAt", transform);
  }

  void switchBuffer()
  {
    activeBuffer = (activeBuffer + 1) % 2;
  }

  int getBufferIndex() { return activeBuffer; }

  int getNextBufferIndex() { return (activeBuffer + 1) % 2; }

private:
  GLuint mainPboIds[2];
  GLuint heightMap;
  GLuint mainVaoIds[2];
  GLuint mainVboIds[2];
  GLuint mainEbo;
  GLuint mainFBO;
  GLuint mainDepthMap;
  GLuint mainNormalIntensityMap;
  std::unique_ptr<Shader> mainShader;

  constexpr static unsigned int SHADOW_WIDTH = 4096;
  constexpr static unsigned int SHADOW_HEIGHT = 4096;
  GLuint shadowDepthMapFBO;
  GLuint shadowDepthMap;
  const glm::mat4 lightProjection = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.4f, 1.5f);
  const glm::mat4 lightView = glm::lookAt(glm::vec3(-0.4472135955, 0.894427191, 0.0),
                                          glm::vec3(0.0f, 0.0f, 0.0f),
                                          glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::mat4 lightSpaceMatrix = lightProjection * lightView;
  std::unique_ptr<Shader> depthShader;

  GLuint postVao;
  GLuint postVbo;
  std::unique_ptr<Shader> postShader;

  void switchTexture(int index)
  {
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, heightMap);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mainPboIds[index]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, 0);
  }

  void redrawImage(int index)
  {
    // depth pass
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    depthShader->use();
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, heightMap);

    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowDepthMapFBO);
      glClear(GL_DEPTH_BUFFER_BIT);
      glBindVertexArray(mainVaoIds[index]);
      glDrawElements(GL_TRIANGLES, 3 * 2 * (height - 1) * (width - 1), GL_UNSIGNED_INT, 0);

    // main pass
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, mainFBO);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glEnable(GL_MULTISAMPLE);

      mainShader->use();
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, heightMap);
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, shadowDepthMap);

      glBindVertexArray(mainVaoIds[index]);
      glDrawElements(GL_TRIANGLES, 3 * 2 * (height - 1) * (width - 1), GL_UNSIGNED_INT, 0);

      glDisable(GL_MULTISAMPLE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // postprocessing pass
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT);
    postShader->use();

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, mainDepthMap);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, mainNormalIntensityMap);

    glBindVertexArray(postVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

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
      mainShader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
    if (glfwGetKey(windowPtr, GLFW_KEY_S) == GLFW_PRESS)
    {
      camera.ProcessKeyboard(BACKWARD, deltaTime);
      mainShader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
    if (glfwGetKey(windowPtr, GLFW_KEY_A) == GLFW_PRESS)
    {
      camera.ProcessKeyboard(LEFT, deltaTime);
      mainShader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
    if (glfwGetKey(windowPtr, GLFW_KEY_D) == GLFW_PRESS)
    {
      camera.ProcessKeyboard(RIGHT, deltaTime);
      mainShader->setVec3("viewPos", camera.front);
      needsRedraw = true;
    }
  }
};
