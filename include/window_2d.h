#pragma once

#include <cmath>
#include <iostream>
#include <memory>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "shader.h"

class Window2D {
public:
  GLFWwindow *window_ptr;

  float *h_cuda_buffers[2];
  cudaGraphicsResource *cudaPboResources[2];

  double c_re = 0.0;
  double c_im = 0.0;
  double theta = 0.0;

  int width;
  int height;

  double zoomLevel = 0.5;
  double x_offset = 0.0;
  double y_offset = 0.0;

  bool tracking_mouse = false;
  double last_mouse_x = 0.0;
  double last_mouse_y = 0.0;

  int active_buffer = 0;
  bool needs_redraw = true;
  bool needs_texture_switch = false;
  bool paused = false;

  Window2D(int width, int height);

  ~Window2D();

  void update_state() {
    update_theta();
    update_pan();

    if (needs_redraw) {
      update_c();
      active_buffer = (active_buffer + 1) % 2;
    }
  }

  void redraw() {
    glfwMakeContextCurrent(window_ptr);
    if (needs_redraw) {
      switch_texture(active_buffer);
      redraw_image();

      needs_redraw = false;
      needs_texture_switch = true;
    } else if (needs_texture_switch) {
      int next_buffer = (active_buffer + 1) % 2;
      switch_texture(next_buffer);
      redraw_image();

      needs_texture_switch = false;
    }
  }

  int getNextBufferIndex() { return (active_buffer + 1) % 2; }

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

  void update_c() {
    c_re = (R - r) * cos(theta) + d * cos((R - r) * theta / r);
    c_im = (R - r) * sin(theta) - d * sin((R - r) * theta / r);
  }

  void update_pan() {
    if (tracking_mouse) // TODO: incorporate into mouse callback?
    {
      double xPos, yPos;
      glfwGetCursorPos(window_ptr, &xPos, &yPos);
      x_offset -= (last_mouse_x - xPos) / width / zoomLevel * 2;
      y_offset += (last_mouse_y - yPos) / height / zoomLevel * 2;

      last_mouse_x = xPos;
      last_mouse_y = yPos;
      needs_redraw = true;
    }
  }

  void update_theta() {
    if (!paused) {
      theta += 0.001;
      needs_redraw = true;
    } else if (glfwGetKey(window_ptr, GLFW_KEY_LEFT) == GLFW_PRESS) {
      theta -= 0.001 / zoomLevel;
      needs_redraw = true;
    } else if (glfwGetKey(window_ptr, GLFW_KEY_RIGHT) == GLFW_PRESS) {
      theta += 0.001 / zoomLevel;
      needs_redraw = true;
    }
  }

  void switch_texture(int index) {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, 0);
  }

  void redraw_image() {
    glClear(GL_COLOR_BUFFER_BIT);
    shader->use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window_ptr);
  }
};
