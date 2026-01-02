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

#include "cnn.h"
#include "mlp.h"
#include "mlp_constants.h"
#include "xgb.h"

class Window2D {
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

  int labelSize = 224;
  float *dLabelImage;
  float *hLabelImage;

  Window2D(int width, int height, GLFWwindow *windowPtr);

  ~Window2D();

  void updateState();

  void redraw(bool switchTex = true);

  inline void switchBuffer() { activeBuffer = (activeBuffer + 1) % 2; }

  inline int getBufferIndex() { return activeBuffer; }

  inline int getNextBufferIndex() { return (activeBuffer + 1) % 2; }

  inline void swap() { glfwSwapBuffers(windowPtr); }

  void updatePrecision();

  inline bool spSufficient() { return singlePrecision; }

private:
  GLuint pboIds[2];
  GLuint texture;
  GLuint vao;
  GLuint vbo;
  std::unique_ptr<Shader> shader;

  GLuint cFbo;
  GLuint cTex;
  GLuint cVao;
  GLuint cVbo;
  std::unique_ptr<Shader> cPointShader;
  std::unique_ptr<Shader> cFadeShader;

  static constexpr double R = 1.7320508075688772; // sqrt(3)
  static constexpr double r = 2.2;
  static constexpr double d = 0.3;
  static constexpr double length = 0.7885;
  static constexpr int cPlotSize = 240;

  double lastThetaUpdate = 0.001;
  bool singlePrecision = true;

  CNNModel cnn = CNNModel("cnn.engine", "cnn.onnx");

  inline void updateC() {
    // if (needsRedraw) {
    //   c.real(sin(sqrt(2.0) * theta) - 0.5);
    //   c.imag(sin(theta));
    //   // c.real((R - r) * cos(theta) + d * cos((R - r) * theta / r));
    //   // c.imag((R - r) * sin(theta) - d * sin((R - r) * theta / r));
    //   // c.real(length * cos(theta));
    //   // c.imag(length * sin(theta));
    // }
    if (paused) {
      if (glfwGetKey(windowPtr, GLFW_KEY_UP) == GLFW_PRESS) {
        c.imag(c.imag() + 0.001 / (zoomLevel * zoomLevel));
        needsRedraw = true;
      }
      if (glfwGetKey(windowPtr, GLFW_KEY_DOWN) == GLFW_PRESS) {
        c.imag(c.imag() - 0.001 / (zoomLevel * zoomLevel));
        needsRedraw = true;
      }
      if (glfwGetKey(windowPtr, GLFW_KEY_LEFT) == GLFW_PRESS) {
        c.real(c.real() - 0.001 / (zoomLevel * zoomLevel));
        needsRedraw = true;
      }
      if (glfwGetKey(windowPtr, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        c.real(c.real() + 0.001 / (zoomLevel * zoomLevel));
        needsRedraw = true;
      }
    }
    glBindBuffer(GL_ARRAY_BUFFER, cVbo);
    float tmp[] = {(float)c.real() + 0.5f, (float)c.imag()};
    glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float), tmp, GL_DYNAMIC_DRAW);
  }

  void updatePan();

  void updateTheta();

  void incrementTheta(double increment);

  void switchTexture(int index);

  void redrawImage();
};
