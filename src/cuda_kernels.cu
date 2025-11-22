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

__device__ uchar3 map_color(float intensity) {
    uchar3 rgb = {0, 0, 0};
    if (intensity < MAX_ITERS) {
      rgb.x = (unsigned char)(__sinf(intensity * 0.05f + 5.423f) * 127 + 128);
      rgb.y = (unsigned char)(__sinf(intensity * 0.05f + 4.359f) * 127 + 128);
      rgb.z = (unsigned char)(__sinf(intensity * 0.05f + 1.150f) * 127 + 128);
    }
    return rgb;
}

__global__ void julia(uchar3 *const buffer, const float range,
                      const float2 offsets, const float2 c, const int width,
                      const int height)
{
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_idx < width && y_idx < height)
  {
    float re = ((float)x_idx / (width - 1)) * range * 2 - range - offsets.x;
    float im = ((float)y_idx / (height - 1)) * range * 2 - range - offsets.y;
    float2 z{re, im};

    float intensity = evaluate(z, c);
    uchar3 rgb = map_color(intensity);

    buffer[y_idx * width + x_idx] = rgb;
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

void compute_julia_cuda(ProgramState state, unsigned char *buffer)
{
  float2 c = make_float2(state.c_re, state.c_im);
  float2 offsets = make_float2(state.x_offset, state.y_offset);
  uchar3 *buffer_ptr = reinterpret_cast<uchar3 *>(buffer);

  unsigned int n_blocks = (state.width + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
  dim3 block_dims{BLOCK_SIZE_2D, BLOCK_SIZE_2D};
  dim3 grid_dims{n_blocks, n_blocks};
  julia<<<grid_dims, block_dims>>>(buffer_ptr, (float)(1.0 / state.zoomLevel),
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
