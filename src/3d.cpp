#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <cstddef>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "avx_kernels.h"
#include "gl_utils.h"

#ifdef WIN32
#include <windows.h>
extern "C"
{
  __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

void smooth_surface(std::vector<float> &data, int width, int height, int radius)
{
  std::vector<float> temp = data;

  std::vector<int> rows(height);
  std::iota(rows.begin(), rows.end(), 0); // 0, 1, 2... height-1

  // Horizontal Pass
  std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&](int y)
                {
        for (int x = 0; x < width; ++x) {
            float sum = 0;
            int count = 0;
            // Sum neighbors (Clamp to edges)
            for (int k = -radius; k <= radius; ++k) {
                int nx = std::clamp(x + k, 0, width - 1);
                sum += data[y * width + nx];
                count++;
            }
            temp[y * width + x] = sum / count;
        } });

  // Vertical Pass
  std::vector<int> cols(width);
  std::iota(cols.begin(), cols.end(), 0);

  std::for_each(std::execution::par_unseq, cols.begin(), cols.end(), [&](int x)
                {
        for (int y = 0; y < height; ++y) {
            float sum = 0;
            int count = 0;
            for (int k = -radius; k <= radius; ++k) {
                int ny = std::clamp(y + k, 0, height - 1);
                sum += temp[ny * width + x];
                count++;
            }
            data[y * width + x] = sum / count;
        } });
}

int main()
{
  // set initial state
  ProgramState state;
  state.width = 1024;
  state.height = 1024;

  // width must be multiple of 8 for avx kernel to work
  state.width = (state.width + 7) / 8 * 8;

  // init gl and window
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(state.width, state.height, "Julia", NULL, NULL);
  if (window == NULL)
  {
    std::cout << "Failed to create window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // set user input callbacks
  glfwSetWindowUserPointer(window, &state);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetKeyCallback(window, key_callback);

  glViewport(0, 0, state.width, state.height);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // init vao to draw on
  int width = 4096;
  int height = 4096;

  std::vector<float> vertices;
  vertices.reserve(3 * width * height);
  std::vector<unsigned int> indices;
  indices.reserve(3 * 2 * (width - 1) * (height - 1));
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      vertices.push_back((float)x / (width - 1) * 2 - 1);
      vertices.push_back((float)y / (height - 1) * 2 - 1);
      vertices.push_back(0.0f);

      if (x < width - 1 && y < height - 1)
      {
        unsigned int i = y * width + x;
        indices.push_back(i);
        indices.push_back(i + 1);
        indices.push_back(i + width + 1);

        indices.push_back(i);
        indices.push_back(i + width);
        indices.push_back(i + width + 1);
      }
    }
  }

  std::vector<float> intensities(width * height);
  julia(intensities.data(), 2.0, 0.0, 0.0, -0.4, 0.6, width, height);
  std::transform(std::execution::par_unseq, intensities.begin(), intensities.end(), intensities.begin(), [](float n)
                 { return n * 0.002f; });

  smooth_surface(intensities, width, height, 20);

  for (int i = 0; i < width * height; ++i)
  {
    vertices[i * 3 + 2] = intensities[i];
  }

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  unsigned int EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // init shaders
  Shader shader("shaders/shader_3d.vs", "shaders/shader_3d.fs");
  shader.use();

  glEnable(GL_DEPTH_TEST);

  unsigned int heightMapTexture;
  glGenTextures(1, &heightMapTexture);
  glBindTexture(GL_TEXTURE_2D, heightMapTexture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, intensities.data());

  glm::mat4 model = glm::mat4(1.0f);
  model = glm::scale(model, glm::vec3(1.0/sqrt(2.0), 1.0, 1.0/sqrt(2.0)));
  glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)state.width / (float)state.height, 0.1f, 100.0f);

  // main render loop
  while (!glfwWindowShouldClose(window))
  {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    shader.use();

    glm::mat4 view = glm::mat4(1.0f);
    float radius = 1.5f;
    float camX = static_cast<float>(sin(glfwGetTime() * 0.5) * radius);
    float camZ = static_cast<float>(cos(glfwGetTime() * 0.5) * radius);
    view = glm::lookAt(glm::vec3(camX, 1.0f, camZ), glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    shader.setMat4("lookAt", projection * view * model);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();
  return 0;
}
