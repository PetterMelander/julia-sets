#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "window_2d.h"

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
  glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
  Window2D *window_2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));

  window_2d->last_mouse_x = xposIn;
  window_2d->last_mouse_y = yposIn;
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
  Window2D *window_2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));

  double oldZoom = window_2d->zoomLevel;
  if (yoffset > 0)
  {
    window_2d->zoomLevel *= 1.1;
  }
  else
  {
    window_2d->zoomLevel /= 1.1;
  }

  double xPos, yPos;
  glfwGetCursorPos(window, &xPos, &yPos);
  window_2d->x_offset -= (1.0 - 1.0 / 1.1) * (xPos / window_2d->width - 0.5) * 2.0 / oldZoom;
  window_2d->y_offset += (1.0 - 1.0 / 1.1) * (yPos / window_2d->height - 0.5) * 2.0 / oldZoom;

  window_2d->needs_redraw = true;
}

void mouse_button_callback(GLFWwindow *window, int button, int action,
                           int mods)
{
  Window2D *window_2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
  {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    window_2d->tracking_mouse = true;
    window_2d->last_mouse_x = xPos;
    window_2d->last_mouse_y = yPos;
    window_2d->needs_redraw = true;
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
  {
    window_2d->tracking_mouse = false;
  }
}

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
  {
    glfwSetWindowShouldClose(window, true);
  }
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
  {
    Window2D *window_2d = static_cast<Window2D *>(glfwGetWindowUserPointer(window));
    window_2d->paused = !window_2d->paused;
  }
}
