#include "gl_utils.h"
#include "shader.h"
#include "cuda_kernels.cuh"
#include "avx_kernels.h"

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
  glViewport(0, 0, width, height);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
  ProgramState *state =
      static_cast<ProgramState *>(glfwGetWindowUserPointer(window));

  double oldZoom = state->zoomLevel;
  if (yoffset > 0)
  {
    state->zoomLevel *= 1.1;
  }
  else
  {
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
                           int mods)
{
  ProgramState *state =
      static_cast<ProgramState *>(glfwGetWindowUserPointer(window));

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
  {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    state->tracking_mouse = true;
    state->last_mouse_x = xPos;
    state->last_mouse_y = yPos;
    state->needs_redraw = true;
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
  {
    state->tracking_mouse = false;
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
    ProgramState *state =
        static_cast<ProgramState *>(glfwGetWindowUserPointer(window));
    state->paused = !state->paused;
  }
}

void update_pan(ProgramState &state, GLFWwindow *window)
{
  if (state.tracking_mouse)
  {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    state.x_offset -=
        (state.last_mouse_x - xPos) / state.width / state.zoomLevel * 2;
    state.y_offset +=
        (state.last_mouse_y - yPos) / state.height / state.zoomLevel * 2;

    state.last_mouse_x = xPos;
    state.last_mouse_y = yPos;
    state.needs_redraw = true;
  }
}

void update_theta(ProgramState &state, GLFWwindow *window)
{
  if (!state.paused)
  {
    state.theta += 0.001;
    state.needs_redraw = true;
  }
  else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
  {
    state.theta -= 0.001 / state.zoomLevel;
    state.needs_redraw = true;
  }
  else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
  {
    state.theta += 0.001 / state.zoomLevel;
    state.needs_redraw = true;
  }
}

void redraw_image(GLFWwindow *window, Shader shader, unsigned int texture, unsigned int VAO)
{
  glClear(GL_COLOR_BUFFER_BIT);

  shader.use();

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);

  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

  glfwSwapBuffers(window);
}

void switch_texture(ProgramState &state, int index, unsigned int texture,
                    GLuint *pboIds)
{
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, state.width, state.height, GL_RGB,
                  GL_UNSIGNED_BYTE, 0);
}

void compute_julia_sp(ProgramState &state, cudaGraphicsResource *cudaPboResource, cudaStream_t stream)
{
  unsigned char *d_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_buffer, nullptr, cudaPboResource));
  compute_julia_cuda(state, d_buffer, stream);
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, stream));
}

void compute_julia_dp(ProgramState &state, float *h_cuda_buffer, float *d_cuda_buffer,
                      cudaGraphicsResource *cudaPboResource, cudaStream_t stream)
{
  compute_julia_avx(state, h_cuda_buffer);
  cudaMemcpyAsync(d_cuda_buffer, h_cuda_buffer, state.width * state.height * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
  unsigned char *d_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_buffer, nullptr, cudaPboResource));

  map_colors_cuda(d_buffer, d_cuda_buffer, state.width * state.height, stream);
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, stream));
}
