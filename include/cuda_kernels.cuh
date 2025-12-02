#pragma once

#include <cstdio>

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

void compute_julia_cuda(int width, int height, double c_re, double c_im, double zoomLevel,
                        double x_offset, double y_offset, float *buffer, cudaStream_t stream);
// void compute_normals_cuda(const ProgramState &state, float *const h, float *out, cudaStream_t stream);
