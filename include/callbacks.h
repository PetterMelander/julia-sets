#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "labeling.h"
#include "window_2d.h"

inline void framebufferSizeCallback(GLFWwindow *window, int width, int height) // TODO: fix or remove
{
  glViewport(0, 0, width, height);
}

inline void mouseCallback(GLFWwindow *window, double xposIn, double yposIn) // TODO: unused
{
  Window2D *window2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));

  window2d->lastMouseX = xposIn;
  window2d->lastMouseY = yposIn;
}

inline void scrollCallback(GLFWwindow *window, double xOffset, double yOffset)
{
  Window2D *window2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));

  window2d->updatePrecision();

  double oldZoom = window2d->zoomLevel;
  if (yOffset > 0.0)
  {
    window2d->zoomLevel *= 1.1;
  }
  else
  {
    window2d->zoomLevel /= 1.1;
  }

  double xPos, yPos;
  glfwGetCursorPos(window, &xPos, &yPos);
  int width = window2d->width;
  int height = window2d->height;
  int minDim = std::min(width, height);
  float xScale = (float)width / minDim;
  float yScale = (float)height / minDim;
  window2d->xOffset -= (1.0 - 1.0 / 1.1) * (xPos / window2d->width - 0.5) * xScale * 2.0 / oldZoom;
  window2d->yOffset += (1.0 - 1.0 / 1.1) * (yPos / window2d->height - 0.5) * yScale * 2.0 / oldZoom;

  window2d->needsRedraw = true;
}

inline void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  Window2D *window2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
  {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    window2d->trackingMouse = true;
    window2d->lastMouseX = xPos;
    window2d->lastMouseY = yPos;
    window2d->needsRedraw = true;
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
  {
    window2d->trackingMouse = false;
  }
}

inline void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
  {
    glfwSetWindowShouldClose(window, true);
  }
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
  {
    Window2D *window2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));
    window2d->paused = !window2d->paused;
  }
  if (key == GLFW_KEY_1)
  {
    Window2D *window2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));
    saveImage(window2d, true);
  }
  if (key == GLFW_KEY_2)
  {
    Window2D *window2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));
    saveImage(window2d, false);
  }
}
