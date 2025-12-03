#pragma once

#include <complex>
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

void computeJuliaCuda(int width, int height, std::complex<double> c, double zoomLevel,
                        double xOffset, double yOffset, float *buffer, cudaStream_t stream);
void computeNormalsCuda(int width, int height, float *const h, float *out, cudaStream_t stream);
