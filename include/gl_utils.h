#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "camera.h"
#include "shader.h"

struct ProgramState
{
  double c_re = 0.0;
  double c_im = 0.0;
  double theta = 0.0;

  int width = 512;
  int height = 512;

  double zoomLevel = 0.5;
  double x_offset = 0.0;
  double y_offset = 0.0;
  bool tracking_mouse = false;
  double last_mouse_x = 0.0;
  double last_mouse_y = 0.0;
  bool needs_redraw = true;
  bool paused = false;

  Camera camera;
};

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void update_pan(ProgramState &state, GLFWwindow *window);
void process_movement(GLFWwindow *window, float deltaTime);
void update_theta(ProgramState &state, GLFWwindow *window);
void redraw_image(GLFWwindow *window, Shader &shader, unsigned int texture, unsigned int VAO);
void switch_texture(const ProgramState &state, int index, unsigned int texture, GLuint *pboIds);
void compute_julia_sp(const ProgramState &state, cudaGraphicsResource *cudaPboColor,
                      cudaGraphicsResource *cudaPboSmoothed, cudaGraphicsResource *cudaVbo, cudaStream_t stream);
void compute_julia_dp(const ProgramState &state, float *h_cuda_buffer, cudaGraphicsResource *cudaPboColor,
                      cudaGraphicsResource *cudaPboSmoothed, cudaGraphicsResource *cudaVbo, cudaStream_t stream);

void redraw_image_3d(GLFWwindow *window, Shader &shader, unsigned int texture, unsigned int VAO);
void switch_texture_3d(const ProgramState &state, int index, unsigned int texture, GLuint *pboIds);
