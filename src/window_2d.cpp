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

#ifndef NDEBUG
void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id,
                            GLenum severity, GLsizei length,
                            const char *message, const void *userParam)
{
  // Ignore non-significant error/warning codes
  if (id == 131169 || id == 131185 || id == 131218 || id == 131204)
    return;

  std::cout << "---------------" << std::endl;
  std::cout << "Debug message (" << id << "): " << message << std::endl;

  switch (source)
  {
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

  switch (type)
  {
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
#endif

Window2D::Window2D(int width, int height, GLFWwindow *windowPtr)
    : windowPtr(windowPtr), width(width), height(height)
{

#ifndef NDEBUG
  int flags;
  glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
  if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
  {
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); // Makes sure errors are displayed at
                                           // the moment they happen
    glDebugMessageCallback(glDebugOutput, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr,
                          GL_TRUE);
  }
#endif

  // allocate buffers, texture, etc.
  int dSize = width * height * sizeof(float);
  glGenBuffers(2, pboIds);
  for (int i = 0; i < 2; ++i)
  {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));

    CUDA_CHECK(cudaMallocHost(&hCudaBuffers[i], dSize));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[i]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, dSize, 0, GL_DYNAMIC_DRAW);

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

  float quadVertices[] = {
      -1.0f,
      1.0f,
      0.0f,
      1.0f,
      -1.0f,
      -1.0f,
      0.0f,
      0.0f,
      1.0f,
      1.0f,
      1.0f,
      1.0f,
      1.0f,
      -1.0f,
      1.0f,
      0.0f,
  };
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glGenFramebuffers(1, &cFbo);
  glBindFramebuffer(GL_FRAMEBUFFER, cFbo);
  glGenTextures(1, &cTex);
  glBindTexture(GL_TEXTURE_2D, cTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cPlotSize, cPlotSize, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         cTex, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glGenVertexArrays(1, &cVao);
  glBindVertexArray(cVao);
  glGenBuffers(1, &cVbo);
  glBindBuffer(GL_ARRAY_BUFFER, cVbo);
  glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glPointSize(2.5f);

  // set window parameters and callbacks
  glfwSwapInterval(0);
  glfwSetWindowUserPointer(windowPtr, this);

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  glfwSetScrollCallback(windowPtr, scrollCallback);
  glfwSetMouseButtonCallback(windowPtr, mouseButtonCallback);
  glfwSetKeyCallback(windowPtr, keyCallback);
  glfwSetFramebufferSizeCallback(windowPtr, framebufferSizeCallback);

  // init shader
  shader =
      std::make_unique<Shader>("shaders/shader_2d.vs", "shaders/shader_2d.fs");
  shader->use();
  shader->setInt("juliaTexRaw", 0);

  cPointShader = std::make_unique<Shader>("shaders/shader_point.vs",
                                          "shaders/shader_point.fs");
  cPointShader->use();
  cPointShader->setFloat("speed", 0.0f);
  cFadeShader = std::make_unique<Shader>("shaders/shader_fade.vs",
                                         "shaders/shader_fade.fs");
  cFadeShader->use();
  cFadeShader->setFloat("increment", 0.0f);

  // init Npp vars
  CUDA_CHECK(cudaMalloc(&dUpdateRelativeError, sizeof(Npp64f)));
  CUDA_CHECK(cudaMallocHost(&hUpdateRelativeError, sizeof(double)));

  manualUpdateC();
  float *tex1 = nullptr;
  float *tex2 = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(2, cudaPboResources, streams[0]));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&tex1, nullptr,
                                                  cudaPboResources[0]));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&tex2, nullptr,
                                                  cudaPboResources[1]));
  computeJuliaCuda(width, height, c, zoomLevel, xOffset, yOffset, tex1,
                   streams[0]);
  computeJuliaCuda(width, height, c, zoomLevel, xOffset, yOffset, tex2,
                   streams[0]);
  CUDA_CHECK(cudaGraphicsUnmapResources(2, cudaPboResources, streams[0]));
  CUDA_CHECK(cudaStreamSynchronize(streams[0]));

  hLabelImage = new float[labelSize * labelSize];
  CUDA_CHECK(cudaMalloc(&dLabelImage, labelSize * labelSize * sizeof(float)));
}

Window2D::~Window2D()
{
  CUDA_CHECK(cudaFreeHost(hCudaBuffers[0]));
  CUDA_CHECK(cudaFreeHost(hCudaBuffers[1]));
  CUDA_CHECK(cudaFree(dUpdateRelativeError));
  CUDA_CHECK(cudaFreeHost(hUpdateRelativeError));

  CUDA_CHECK(cudaStreamDestroy(streams[0]));
  CUDA_CHECK(cudaStreamDestroy(streams[1]));

  CUDA_CHECK(cudaFree(dLabelImage));
  delete[] hLabelImage;
}

void Window2D::updateState()
{
  CUDA_CHECK(cudaStreamSynchronize(streams[(activeBuffer + 1) % 2]));
  updateTheta();
  updatePan();
  manualUpdateC();
}

void Window2D::redraw(bool switchTex)
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

void Window2D::updatePrecision()
{
  float minDim = std::min(width, height);
  float newZoom = zoomLevel * powf(1.1f, 5.0f) * minDim / cnn.imgSize;

  if (newZoom < 300.0f)
  {
    singlePrecision = true;
    return;
  }
  else if (newZoom > 100000.0f)
  {
    singlePrecision = false;
    return;
  }

  float result = 0;

  // enqueue cnn pred
  cnn.enqueue(c, newZoom, xOffset, yOffset);

  // xgb pred
  std::vector<Entry> entries(5);
  entries[0].fvalue = ((float)c.real() - INPUT_MEANS[0]) / INPUT_STDS[0];
  entries[1].fvalue = ((float)c.imag() - INPUT_MEANS[1]) / INPUT_STDS[1];
  entries[2].fvalue = ((float)xOffset - INPUT_MEANS[2]) / INPUT_STDS[2];
  entries[3].fvalue = ((float)yOffset - INPUT_MEANS[3]) / INPUT_STDS[3];
  entries[4].fvalue =
      ((float)log(newZoom + 1.0) - INPUT_MEANS[4]) / INPUT_STDS[4];
  predict(entries.data(), 0, &result);

  // mlp pred
  float windowParams[] = {entries[0].fvalue, entries[1].fvalue,
                          entries[2].fvalue, entries[3].fvalue,
                          entries[4].fvalue};
  result += 1.0f / (1.0f + expf(-mlpPredict(windowParams)));

  // get cnn pred
  result += 2.0f * 1.0f / (1.0f + expf(-cnn.getPred()));

  // weight ensemble 1 - 1 - 2
  result /= 4.0f;

  singlePrecision = result >= 0.5f;
}

void Window2D::updatePan()
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

void Window2D::updateTheta()
{
  if (!paused)
  {
    double thetaUpdate = 0.7 * lastThetaUpdate +
                         std::min(0.001, exp(-*hUpdateRelativeError * 10000)) +
                         0.00001;
    double gradientSize =
        sqrt(pow(sqrt(2.0) * cos(theta), 2.0) + pow(cos(theta), 2.0));
    thetaUpdate = std::min(lastThetaUpdate * 2.0, thetaUpdate);
    lastThetaUpdate = thetaUpdate;
    incrementTheta(thetaUpdate / std::max(gradientSize, 0.1));
  }
  else if (glfwGetKey(windowPtr, GLFW_KEY_COMMA) == GLFW_PRESS)
  {
    incrementTheta(-0.001 / zoomLevel);
  }
  else if (glfwGetKey(windowPtr, GLFW_KEY_PERIOD) == GLFW_PRESS)
  {
    incrementTheta(0.001 / zoomLevel);
  }
}

void Window2D::incrementTheta(double increment)
{
  theta += increment;
  c.real(sin(sqrt(2.0) * theta) - 0.5);
  c.imag(sin(theta));
  cFadeShader->use();
  cFadeShader->setFloat("increment", 1.0f - std::abs(increment) * 0.1);
  cPointShader->use();
  cPointShader->setFloat("speed", 1.0f - std::abs(increment) * 200);
  needsRedraw = true;
}

void Window2D::switchTexture(int index)
{
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, 0);
}

void Window2D::redrawImage()
{
  // draw julia set
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  glViewport(0, 0, width, height);
  shader->use();

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);

  glBindVertexArray(vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // fade c plot
  glViewport(0, 0, cPlotSize, cPlotSize);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, cFbo);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, cFbo);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ZERO, GL_SRC_ALPHA);
  cFadeShader->use();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // add new c point
  cPointShader->use();
  glBindVertexArray(cVao);
  glDrawArrays(GL_POINTS, 0, 1);
  glDisable(GL_BLEND);

  // blit c plot to screen
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glBlitFramebuffer(0, 0, cPlotSize, cPlotSize, 0, height - cPlotSize,
                    cPlotSize, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}
