#pragma once

#include <cmath>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cuda_kernels.cuh"

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

  Window2D(int width, int height) : width(width), height(height) {

    // create window
    glfwWindowHint(GLFW_SAMPLES, 0);
    window_ptr = glfwCreateWindow(width, height, "Julia", NULL, NULL);
    if (window_ptr == NULL) {
      std::cout << "Failed to create 2D window" << std::endl;
      glfwTerminate();
    }
    glfwMakeContextCurrent(window_ptr);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      std::cout << "Failed to initialize GLAD" << std::endl;
    }

    // allocate buffers, texture, etc.
    int dsize = width * height * sizeof(float);
    glGenBuffers(2, pboIds);
    for (int i = 0; i < 2; ++i) {

      cudaMallocHost(&h_cuda_buffers[i], dsize);

      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[i]);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, dsize, 0, GL_DYNAMIC_DRAW);

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

    float vertices[] = {1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
                        -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f,  0.0f, 1.0f};
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    unsigned int indices[] = {0, 1, 3, 1, 2, 3};
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // set window parameters and callbacks
    glfwSwapInterval(0);
    glfwSetWindowUserPointer(window_ptr, this);

    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glfwSetScrollCallback(window_ptr, scroll_callback);
    glfwSetMouseButtonCallback(window_ptr, mouse_button_callback);
    glfwSetKeyCallback(window_ptr, key_callback);
    glfwSetFramebufferSizeCallback(window_ptr, framebuffer_size_callback);

    // init shader
    shader.use();
    shader.setInt("texture1", 0);
  }

  ~Window2D() {
    cudaFreeHost(h_cuda_buffers[0]);
    cudaFreeHost(h_cuda_buffers[1]);
  }

  void redraw() {
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

private:
  GLuint pboIds[2];
  GLuint texture;
  GLuint vao;
  GLuint vbo;
  GLuint ebo;

  Shader shader{"shaders/shader.vs", "shaders/shader.fs"};

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
    shader.use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window_ptr);
  }

  void update_state() {
    update_theta();
    update_pan();

    if (needs_redraw) {
      update_c();
      active_buffer = (active_buffer + 1) % 2;
    }
  }
};