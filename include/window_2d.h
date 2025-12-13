#pragma once

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cuda_kernels.cuh"
#include "shader.h"

class Window2D
{
public:
  GLFWwindow *windowPtr;

  float *hCudaBuffers[2];
  cudaGraphicsResource *cudaPboResources[2];
  cudaStream_t streams[2];

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

  Window2D(int width, int height);

  ~Window2D();

  void updateState()
  {
    updateTheta();
    updatePan();

    if (needsRedraw)
    {
      updateC();
    }
  }

  void redraw()
  {
    CUDA_CHECK(cudaStreamSynchronize(streams[activeBuffer]));

    glfwMakeContextCurrent(windowPtr);
    switchTexture(activeBuffer);
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
    glfwMakeContextCurrent(windowPtr);
    glfwSwapBuffers(windowPtr);
  }

private:
  GLuint pboIds[2];
  GLuint texture;
  GLuint vao;
  GLuint vbo;
  GLuint ebo;

  std::unique_ptr<Shader> shader;

  static constexpr double R = 1.7320508075688772; // sqrt(3)
  static constexpr double r = 2.2;
  static constexpr double d = 0.3;
  static constexpr double length = 0.7885;

  void updateC()
  {
    // c.real((R - r) * cos(theta) + d * cos((R - r) * theta / r));
    // c.imag((R - r) * sin(theta) - d * sin((R - r) * theta / r));
    c.real(length * cos(theta));
    c.imag(length * sin(theta));
  }

  void updatePan()
  {
    if (trackingMouse) // TODO: incorporate into mouse callback?
    {
      double xPos, yPos;
      glfwGetCursorPos(windowPtr, &xPos, &yPos);
      xOffset -= (lastMouseX - xPos) / width / zoomLevel * 2;
      yOffset += (lastMouseY - yPos) / height / zoomLevel * 2;

      lastMouseX = xPos;
      lastMouseY = yPos;
      needsRedraw = true;
    }
  }

  void updateTheta()
  {
    if (!paused)
    {
      theta += 0.001;
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
    glClear(GL_COLOR_BUFFER_BIT);
    shader->use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // glfwSwapBuffers(windowPtr);
  }
};
