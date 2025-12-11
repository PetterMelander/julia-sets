#pragma once

#include <complex>
#include <cstdio>

#ifdef NDEBUG
#define CUDA_CHECK(call) call
#define NPP_CHECK(call) call
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

#define NPP_CHECK(call)                                                                         \
  do                                                                                            \
  {                                                                                             \
    NppStatus _status = (call);                                                                 \
    if (_status != NPP_SUCCESS)                                                                 \
    {                                                                                           \
      std::cerr << "\n[NPP ERROR] at " << __FILE__ << ":" << __LINE__ << "\n"                   \
                << "   Code: " << _status << "\n";                                              \
                                                                                                \
      if (_status == NPP_CUDA_KERNEL_EXECUTION_ERROR)                                           \
      {                                                                                         \
        cudaError_t _cudaErr = cudaGetLastError();                                              \
        if (_cudaErr != cudaSuccess)                                                            \
        {                                                                                       \
          std::cerr << "   >> Underlying CUDA Error: "                                          \
                    << cudaGetErrorString(_cudaErr)                                             \
                    << " (" << _cudaErr << ")\n";                                               \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
          std::cerr << "   >> CUDA Status is Success (Error might be internal to NPP logic)\n"; \
        }                                                                                       \
      }                                                                                         \
                                                                                                \
      if (_status < 0)                                                                          \
      {                                                                                         \
        std::cerr << "   Process terminating due to NPP error.\n";                              \
        std::exit(EXIT_FAILURE);                                                                \
      }                                                                                         \
      else                                                                                      \
      {                                                                                         \
        std::cerr << "   Warning only. Execution continuing.\n";                                \
      }                                                                                         \
    }                                                                                           \
  } while (0)
#endif

void computeJuliaCuda(int width, int height, std::complex<double> c, double zoomLevel,
                      double xOffset, double yOffset, float *buffer, cudaStream_t stream);
void computeNormalsCuda(int width, int height, float *h, float *out, cudaStream_t stream);
void rescaleImage(int width, int height, float *imgMin, float *imgMax, float *h, cudaStream_t stream);
