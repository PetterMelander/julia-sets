#include <nppi_filtering_functions.h>

#include "avx_kernels.h"
#include "cuda_kernels.cuh"
#include "gl_utils.h"
#include "shader.h"

void update_pan(ProgramState &state, GLFWwindow *window)
{
  if (state.tracking_mouse) // TODO: incorporate into mouse callback?
  {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    state.x_offset -= (state.last_mouse_x - xPos) / state.width / state.zoomLevel * 2;
    state.y_offset += (state.last_mouse_y - yPos) / state.height / state.zoomLevel * 2;

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

void redraw_image(GLFWwindow *window, Shader &shader, unsigned int texture, unsigned int VAO)
{
  glClear(GL_COLOR_BUFFER_BIT);
  shader.use();

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);

  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

  glfwSwapBuffers(window);
}

void switch_texture(const ProgramState &state, int index, unsigned int texture, GLuint *pboIds)
{
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, state.width, state.height, GL_RED, GL_FLOAT, 0);
}

void redraw_image_3d(GLFWwindow *window, Shader &shader, unsigned int texture, unsigned int VAO)
{
  ProgramState *state =
      static_cast<ProgramState *>(glfwGetWindowUserPointer(window));

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_MULTISAMPLE);
  shader.use();

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, texture);

  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, 3 * 2 * (state->height - 1) * (state->width - 1), GL_UNSIGNED_INT, 0);

  glDisable(GL_MULTISAMPLE);

  glfwSwapBuffers(window);
}

void switch_texture_3d(const ProgramState &state, int index, unsigned int texture, GLuint *pboIds)
{
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[index]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, state.width, state.height, GL_RED, GL_FLOAT, 0);
}

void compute_julia_sp(const ProgramState &state, cudaGraphicsResource *cudaPboColor,
                      cudaGraphicsResource *cudaPboSmoothed, cudaGraphicsResource *cudaVbo, cudaStream_t stream)
{
  float *d_color_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboColor, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_color_buffer, nullptr, cudaPboColor));

  float *d_smoothed_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboSmoothed, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_smoothed_buffer, nullptr, cudaPboSmoothed));

  float *d_vbo = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVbo, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, nullptr, cudaVbo));

  // compute_julia_cuda(state, d_color_buffer, stream);

  // smooth for 3d
  NppiSize size = {state.width, state.height};
  NppStreamContext ctx;
  ctx.hStream = stream;
  nppiFilterGaussBorder_32f_C1R_Ctx(
      d_color_buffer,
      sizeof(float) * state.width,
      size,
      NppiPoint{0, 0},
      d_smoothed_buffer,
      sizeof(float) * state.width,
      size,
      NPP_MASK_SIZE_9_X_9,
      NPP_BORDER_REPLICATE,
      ctx);

  // get normals for lighting
  compute_normals_cuda(state, d_smoothed_buffer, d_vbo, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboColor, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboSmoothed, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVbo, stream));
}

void compute_julia_dp(const ProgramState &state, float *h_cuda_buffer, cudaGraphicsResource *cudaPboColor,
                      cudaGraphicsResource *cudaPboSmoothed, cudaGraphicsResource *cudaVbo, cudaStream_t stream)
{
  float *d_color_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboColor, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_color_buffer, nullptr, cudaPboColor));

  float *d_smoothed_buffer = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboSmoothed, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_smoothed_buffer, nullptr, cudaPboSmoothed));

  float *d_vbo = nullptr;
  CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVbo, 0));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, nullptr, cudaVbo));

  // compute the new julia set into 2d texture
  compute_julia_avx(state, h_cuda_buffer);
  CUDA_CHECK(cudaMemcpyAsync(d_color_buffer, h_cuda_buffer, state.width * state.height * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  // smooth for 3d
  NppiSize size = {state.width, state.height};
  NppStreamContext ctx;
  ctx.hStream = stream;
  nppiFilterGaussBorder_32f_C1R_Ctx(
      d_color_buffer,
      sizeof(float) * state.width,
      size,
      NppiPoint{0, 0},
      d_smoothed_buffer,
      sizeof(float) * state.width,
      size,
      NPP_MASK_SIZE_9_X_9,
      NPP_BORDER_REPLICATE,
      ctx);

  // get normals for lighting
  compute_normals_cuda(state, d_smoothed_buffer, d_vbo, stream);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboColor, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboSmoothed, stream));
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVbo, stream));
}

void process_movement(GLFWwindow *window, float deltaTime)
{
  ProgramState *state =
      static_cast<ProgramState *>(glfwGetWindowUserPointer(window));
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
  {
    state->camera.ProcessKeyboard(FORWARD, deltaTime);
    state->needs_redraw = true;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
  {
    state->camera.ProcessKeyboard(BACKWARD, deltaTime);
    state->needs_redraw = true;
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
  {
    state->camera.ProcessKeyboard(LEFT, deltaTime);
    state->needs_redraw = true;
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
  {
    state->camera.ProcessKeyboard(RIGHT, deltaTime);
    state->needs_redraw = true;
  }
}