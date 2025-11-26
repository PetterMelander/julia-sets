#pragma once

#include <cstdio>
#include "gl_utils.h"

#ifdef NDEBUG
#define CUDA_CHECK(call) call
#else
#define CUDA_CHECK(call)                                               \
  do                                                                   \
  {                                                                    \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess)                                            \
    {                                                                  \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)
#endif

void compute_julia_cuda(ProgramState state, float *__restrict__ buffer, cudaStream_t stream);
void map_colors_cuda(unsigned char *__restrict__ buffer, const float *__restrict__ intensities, const int dsize, cudaStream_t stream);
void compute_julia_cuda_smoothed(ProgramState state, int halo, float *buffer, cudaStream_t stream);
