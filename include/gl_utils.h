#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

struct ProgramState {
  double c_re;
  double c_im;

  int width;
  int height;

  double zoomLevel = 0.5;
  double x_offset = 0.0;
  double y_offset = 0.0;
  bool tracking_mouse = false;
  double last_mouse_x = 0;
  double last_mouse_y = 0;
  bool needs_redraw = true;
};

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow *window, int button, int action,
                           int mods);
void process_input(GLFWwindow *window);
void update_pan(ProgramState &state, GLFWwindow *window);
