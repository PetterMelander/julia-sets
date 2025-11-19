#include "gl_utils.h"

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  ProgramState *state =
      static_cast<ProgramState *>(glfwGetWindowUserPointer(window));

  double oldZoom = state->zoomLevel;
  if (yoffset > 0) {
    state->zoomLevel *= 1.1;
  } else {
    state->zoomLevel /= 1.1;
  }

  double xPos, yPos;
  glfwGetCursorPos(window, &xPos, &yPos);
  state->x_offset -=
      (1.0 - 1.0 / 1.1) * (xPos / state->width - 0.5) * 2.0 / oldZoom;
  state->y_offset +=
      (1.0 - 1.0 / 1.1) * (yPos / state->height - 0.5) * 2.0 / oldZoom;

  state->needs_redraw = true;
}

void mouse_button_callback(GLFWwindow *window, int button, int action,
                           int mods) {
  ProgramState *state =
      static_cast<ProgramState *>(glfwGetWindowUserPointer(window));

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    state->tracking_mouse = true;
    state->last_mouse_x = xPos;
    state->last_mouse_y = yPos;
    state->needs_redraw = true;
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
    state->tracking_mouse = false;
  }
}

void update_pan(ProgramState &state, GLFWwindow *window) {
  if (state.tracking_mouse) {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    state.x_offset -=
        (state.last_mouse_x - xPos) / state.width / state.zoomLevel * 2;
    state.y_offset +=
        (state.last_mouse_y - yPos) / state.height / state.zoomLevel * 2;

    state.last_mouse_x = xPos;
    state.last_mouse_y = yPos;
  }
}