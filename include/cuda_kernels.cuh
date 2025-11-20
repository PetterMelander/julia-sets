#pragma once

#include <cstdio>
#include "gl_utils.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void compute_julia_cuda(ProgramState state, unsigned char *buffer);
void map_colors_cuda(uchar3 *__restrict__ buffer, const float *__restrict__ intensities, const int dsize);