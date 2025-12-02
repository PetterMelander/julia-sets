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

class Window3D {
public:
  GLFWwindow *window_ptr;

  cudaGraphicsResource *cudaPboResources[2];
  cudaGraphicsResource *cudaVboResources[2];

  int width;
  int height;

  int active_buffer = 0;
  bool needs_redraw = true;
  bool needs_texture_switch = false;

  Camera camera;

  Window3D(int width, int height);

  void switchBuffer() { active_buffer = (active_buffer + 1) % 2; }

  void redraw(bool switchBuffer) {
    if (switchBuffer) {
      active_buffer = (active_buffer + 1) % 2;
    }

    glfwMakeContextCurrent(window_ptr);
    switch_texture(active_buffer);
    redraw_image(active_buffer);
  }

  int getNextBufferIndex() { return (active_buffer + 1) % 2; }

private:
  GLuint pboIds[2];
  GLuint texture;
  GLuint vaoIds[2];
  GLuint vboIds[2];
  GLuint ebo;

  glm::mat4 projection;

  std::unique_ptr<Shader> shader;

  void switch_texture(int index) {
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, 0);
  }

  void redraw_image(int index) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_MULTISAMPLE);

    shader->use();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(vaoIds[index]);
    glDrawElements(GL_TRIANGLES, 3 * 2 * (height - 1) * (width - 1),
                   GL_UNSIGNED_INT, 0);

    glDisable(GL_MULTISAMPLE);

    glfwSwapBuffers(window_ptr);
  }
};
