#include "cuda_kernels.cuh"
#include "gl_utils.h"

constexpr int MAX_ITERS = 1000;
constexpr float R = 2.f;
constexpr int BLOCK_SIZE_2D = 16;
constexpr int BLOCK_SIZE_1D = 512;

__device__ float evaluate(float2 z, const float2 c)
{
  float escape_iter = MAX_ITERS;
  float escape_abs2 = 0;
  for (int i = 0; i < MAX_ITERS; ++i)
  {
    float abs2 = z.x * z.x + z.y * z.y;
    if (abs2 >= R * R)
    {
      escape_iter = i;
      escape_abs2 = abs2;
      break;
    }
    float re = z.x * z.x - z.y * z.y + c.x;
    float im = 2 * z.x * z.y + c.y;
    z.x = re;
    z.y = im;
  }
  if (escape_iter < MAX_ITERS)
  {
    return escape_iter + 1 - __logf(__logf(__fsqrt_rn(escape_abs2))) / logf(2);
  }
  return MAX_ITERS;
}

__device__ uchar3 map_color(float intensity)
{
  uchar3 rgb = {0, 0, 0};
  if (intensity < MAX_ITERS)
  {
    rgb.x = (unsigned char)(__sinf(intensity * 0.05f + 5.423f) * 127 + 128);
    rgb.y = (unsigned char)(__sinf(intensity * 0.05f + 4.359f) * 127 + 128);
    rgb.z = (unsigned char)(__sinf(intensity * 0.05f + 1.150f) * 127 + 128);
  }
  return rgb;
}

__global__ void julia(float *const __restrict__ buffer, const float range, const float2 offsets,
                      const float2 c, const int width, const int height)
{
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_idx < width && y_idx < height)
  {
    float re = ((float)x_idx / (width - 1)) * range * 2 - range - offsets.x;
    float im = ((float)y_idx / (height - 1)) * range * 2 - range - offsets.y;
    float2 z{re, im};

    float intensity = evaluate(z, c);
    buffer[y_idx * width + x_idx] = intensity;
  }
}

__global__ void julia_smoothed_y(float *__restrict__ buffer, const int halo, const float range, const float2 offsets,
                                 const float2 c, const int width, const int height)
{
  int x_idx = blockIdx.x;

  int effective_dim = blockDim.x - 2 * halo;
  int y_idx = blockIdx.y * effective_dim + threadIdx.x - halo;

  __shared__ float escape_iters[512];

  float intensity = 0.0f;
  if (y_idx < height + halo)
  {
    float re = ((float)x_idx / (width - 1)) * range * 2 - range - offsets.x;
    float im = ((float)y_idx / (height - 1)) * range * 2 - range - offsets.y;
    float2 z{re, im};

    intensity = evaluate(z, c) * 0.001f;
  }

  escape_iters[threadIdx.x] = intensity;
  __syncthreads();

  if (threadIdx.x >= halo && threadIdx.x <= blockDim.y - halo && y_idx < height)
  {
    float accum = 0.0f;
    for (int i = -halo; i <= halo; ++i)
    {
      accum += escape_iters[threadIdx.x + i];
    }
    accum = accum / (2 * halo + 1);
    buffer[y_idx * width + x_idx] = accum;
  }
}

__global__ void smooth_julia_x(float *buffer, const int halo, const float range, const float2 offsets,
                               const float2 c, const int width, const int height)
{
  int y_idx = blockIdx.y;

  int effective_dim = blockDim.x - 2 * halo;
  int x_idx = blockIdx.x * effective_dim + threadIdx.x - halo;
  __shared__ float escape_iters[512];

  float intensity = 0.0f;
  if (x_idx >= 0 && x_idx < width)
  {
    intensity = buffer[blockIdx.y * width + x_idx];
  }

  escape_iters[threadIdx.x] = intensity;
  __syncthreads();

  if (threadIdx.x >= halo && threadIdx.x <= blockDim.x - halo && x_idx < width)
  {
    float accum = 0.0f;
    for (int i = -halo; i <= halo; ++i)
    {
      accum += escape_iters[threadIdx.x + i];
    }
    accum = accum / (2 * halo + 1);
    buffer[y_idx * width + x_idx] = accum;
  }
  if (x_idx < halo)
  {
    float accum = 0.0f;
    for (int i = -halo; i <= halo; ++i)
    {
      if (i >= 0)
      {
        accum += escape_iters[threadIdx.x + i];
      }
      accum = accum / (halo + 1 + x_idx);
    }
    buffer[y_idx * width + x_idx] = accum;
  }
  else if (x_idx > width - halo && x_idx < width)
  {
    float accum = 0.0f;
    for (int i = -halo; i <= halo; ++i)
    {
      if (i < blockDim.x)
      {
        accum += escape_iters[threadIdx.x + i];
      }
      accum = accum / (halo + 1 + (blockDim.x - 1 - x_idx));
    }
    buffer[y_idx * width + x_idx] = accum;
  }
}

__global__ void map_colors(uchar3 *__restrict__ buffer, const float *__restrict__ intensities, const int dsize)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dsize)
  {
    buffer[idx] = map_color(intensities[idx]);
  }
}

void compute_julia_cuda(ProgramState state, float *__restrict__ buffer, cudaStream_t stream)
{
  float2 c = make_float2(state.c_re, state.c_im);
  float2 offsets = make_float2(state.x_offset, state.y_offset);

  unsigned int n_blocks = (state.width + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
  dim3 block_dims{BLOCK_SIZE_2D, BLOCK_SIZE_2D};
  dim3 grid_dims{n_blocks, n_blocks};
  julia<<<grid_dims, block_dims, 0, stream>>>(buffer, (float)(1.0 / state.zoomLevel),
                                              offsets, c, state.width, state.height);
  CUDA_CHECK(cudaGetLastError());
}

void map_colors_cuda(unsigned char *__restrict__ buffer, const float *__restrict__ intensities, const int dsize, cudaStream_t stream)
{
  uchar3 *buffer_ptr = reinterpret_cast<uchar3 *>(buffer);

  int num_blocks = (dsize + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
  map_colors<<<num_blocks, BLOCK_SIZE_1D, 0, stream>>>(buffer_ptr, intensities, dsize);
  CUDA_CHECK(cudaGetLastError());
}

void compute_julia_cuda_smoothed(ProgramState state, int halo, float *buffer, cudaStream_t stream)
{
  float2 c = make_float2(state.c_re, state.c_im);
  float2 offsets = make_float2(state.x_offset, state.y_offset);

  unsigned int n_blocks = (state.height + (BLOCK_SIZE_1D - 2 * halo) - 1) / (BLOCK_SIZE_1D - 2 * halo);
  dim3 grid_dims{static_cast<unsigned int>(state.width), n_blocks};
  julia_smoothed_y<<<grid_dims, BLOCK_SIZE_1D, 512 * sizeof(float), stream>>>(buffer, halo, (float)(1.0 / state.zoomLevel),
                                                                              offsets, c, state.width, state.height);
  CUDA_CHECK(cudaGetLastError());

  n_blocks = (state.width + (BLOCK_SIZE_1D - 2 * halo) - 1) / (BLOCK_SIZE_1D - 2 * halo);
  grid_dims = dim3{n_blocks, static_cast<unsigned int>(state.height)};
  smooth_julia_x<<<grid_dims, BLOCK_SIZE_1D, 512 * sizeof(float), stream>>>(buffer, halo, (float)(1.0 / state.zoomLevel),
                                                                            offsets, c, state.width, state.height);
  CUDA_CHECK(cudaGetLastError());
}