#pragma once

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <nppcore.h>

#include "cuda_kernels.cuh"
#include "shader.h"

class Window2D
{
public:
  GLFWwindow *windowPtr;

  float *hCudaBuffers[2];
  cudaGraphicsResource *cudaPboResources[2];
  cudaStream_t streams[2];

  Npp64f *dUpdateRelativeError;
  double *hUpdateRelativeError;

  std::complex<double> c;
  double theta = 0.0;

  int width;
  int height;

  double zoomLevel = 0.5;
  double xOffset = 0.0;
  double yOffset = 0.0;

  bool trackingMouse = false;
  double lastMouseX = 0.0;
  double lastMouseY = 0.0;

  int activeBuffer = 0;
  bool needsRedraw = true;
  bool needsTextureSwitch = false;
  bool paused = false;

  Window2D(int width, int height, GLFWwindow *windowPtr);

  ~Window2D();

  void updateState()
  {
    CUDA_CHECK(cudaStreamSynchronize(streams[(activeBuffer + 1) % 2]));
    updateTheta();
    updatePan();

    if (needsRedraw)
    {
      updateC();
    }
  }

  void redraw(bool switchTex = true)
  {
    if (switchTex)
    {
      switchTexture(activeBuffer);
    }
      redrawImage();
    if (needsRedraw)
    {
      needsRedraw = false;
      needsTextureSwitch = true;
    }
    else if (needsTextureSwitch)
    {
      needsTextureSwitch = false;
    }
  }

  void switchBuffer()
  {
    activeBuffer = (activeBuffer + 1) % 2;
  }

  int getBufferIndex() { return activeBuffer; }

  int getNextBufferIndex() { return (activeBuffer + 1) % 2; }

  void swap() {
    glfwSwapBuffers(windowPtr);
  }

private:
  GLuint pboIds[2];
  GLuint texture;
  GLuint vao;
  GLuint vbo;

  std::unique_ptr<Shader> shader;

  static constexpr double R = 1.7320508075688772; // sqrt(3)
  static constexpr double r = 2.2;
  static constexpr double d = 0.3;
  static constexpr double length = 0.7885;

  double lastThetaUpdate = 0;

  void updateC()
  {
    c.real(sin(sqrt(2.0) * theta));
    c.imag(sin(theta));
    // c.real((R - r) * cos(theta) + d * cos((R - r) * theta / r));
    // c.imag((R - r) * sin(theta) - d * sin((R - r) * theta / r));
    // c.real(length * cos(theta));
    // c.imag(length * sin(theta));
  }

  void updatePan()
  {
    if (trackingMouse) // TODO: incorporate into mouse callback?
    {
      double xPos, yPos;
      glfwGetCursorPos(windowPtr, &xPos, &yPos);
      xOffset -= (lastMouseX - xPos) / std::min(width, height) / zoomLevel * 2;
      yOffset += (lastMouseY - yPos) / std::min(width, height) / zoomLevel * 2;

      lastMouseX = xPos;
      lastMouseY = yPos;
      needsRedraw = true;
    }
  }

  void updateTheta()
  {
    if (!paused)
    {
      double thetaUpdate = 0.8 * lastThetaUpdate + std::min(0.001, exp(-*hUpdateRelativeError * 10000)) + 0.00001;
      theta += std::min(lastThetaUpdate * 2.0, thetaUpdate);
      lastThetaUpdate = thetaUpdate;
      needsRedraw = true;
    }
    else if (glfwGetKey(windowPtr, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
      theta -= 0.001 / zoomLevel;
      needsRedraw = true;
    }
    else if (glfwGetKey(windowPtr, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
      theta += 0.001 / zoomLevel;
      needsRedraw = true;
    }
  }

  void switchTexture(int index)
  {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, 0);
  }

  void redrawImage()
  {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    glViewport(0, 0, width, height);
    shader->use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
};
