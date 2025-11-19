#include <cmath>
#include <cstddef>
#include <glad/glad.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <immintrin.h>
#include <mm_malloc.h>
#include <ostream>
#include "shader.h"
#include "stb_image.h"
#include "julia.h"
#include "gl_utils.h"

int main() {

  ProgramState state;
  state.width = 1024;
  state.height = 1024;
  state.c_re = 0.35;
  state.c_im = 0.35;

  unsigned char *buffer = (unsigned char *)_mm_malloc(
      sizeof(unsigned char) * state.width * state.height * 3, 64);

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(state.width, state.height, "Julia", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  glfwSetWindowUserPointer(window, &state);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);

  glViewport(0, 0, state.width, state.height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  unsigned int texture1;
  glGenTextures(1, &texture1);
  glBindTexture(GL_TEXTURE_2D, texture1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, state.width, state.height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, buffer);

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  float vertices[] = {1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
                      -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f,  0.0f, 1.0f};
  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  unsigned int indices[] = {0, 1, 3, 1, 2, 3};
  unsigned int EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  Shader shader("src/shaders/shader.vs", "src/shaders/shader.fs");
  shader.use();
  shader.setInt("texture1", 0);

  double length = 0.7885;
  double theta = 0.0;
  bool paused = false;
  while (!glfwWindowShouldClose(window)) {
    process_input(window);

    update_pan(state, window);

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
      paused = !paused;
    }
    if (!paused) {
      theta += 0.001;
    } else {
      if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        theta -= 0.001 / state.zoomLevel;
        state.needs_redraw = true;
      } else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        theta += 0.001 / state.zoomLevel;
        state.needs_redraw = true;
      }
    }
    if (!paused || state.needs_redraw) {
      state.c_re = length * cos(theta);
      state.c_im = length * sin(theta);
      compute_julia(state, buffer);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, state.width, state.height, GL_RGB,
                    GL_UNSIGNED_BYTE, buffer);

    int mouse_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (mouse_state != GLFW_PRESS) {
      state.needs_redraw = false;
    }

    glClear(GL_COLOR_BUFFER_BIT);

    shader.use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  _mm_free(buffer);
  return 0;
}
