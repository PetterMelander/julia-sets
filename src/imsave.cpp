#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <malloc.h>
#include <sstream>
#include <string>

#include <stb_image_write.h>

#include "avx_kernels.h"
#include "imsave.h"
#include "window_2d.h"

constexpr int width = 3840 * 2, height = 2160 * 2;

static void mapColors(const float *buf, uint8_t *cbuf)
{
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < width * height; ++i)
  {
    float intensity = buf[i];
    if (intensity < 1000000.0f)
    {
      intensity = intensity * 0.05;
      cbuf[3 * i + 0] = (uint8_t)((sinf(intensity + 5.423) * 0.5 + 0.5) * 255);
      cbuf[3 * i + 1] = (uint8_t)((sinf(intensity + 4.359) * 0.5 + 0.5) * 255);
      cbuf[3 * i + 2] = (uint8_t)((sinf(intensity + 1.150) * 0.5 + 0.5) * 255);
    }
    else
    {
      cbuf[3 * i + 0] = (uint8_t)0;
      cbuf[3 * i + 1] = (uint8_t)0;
      cbuf[3 * i + 2] = (uint8_t)0;
    }
  }
}

void saveImage(Window2D *window)
{

  float *buf = (float *)_mm_malloc(width * height * sizeof(float), 64);
  uint8_t *cbuf =
      (uint8_t *)_mm_malloc(width * height * 3 * sizeof(uint8_t), 64);
  computeJuliaAvx(width, height, true, window->c, window->zoomLevel, window->xOffset,
                  window->yOffset, buf);
  mapColors(buf, cbuf);

  // get filename
  std::string imgDir = "../saved_imgs/";
  std::filesystem::create_directories(imgDir);
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << imgDir << now_c << ".png";
  std::string filename = ss.str();

  stbi_write_png(filename.c_str(), width, height, 3, cbuf,
                 width * 3 * sizeof(uint8_t));

  _mm_free(buf);
  _mm_free(cbuf);
}