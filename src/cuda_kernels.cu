#include "cuda_kernels.cuh"

constexpr int BLOCK_SIZE_JULIA = 16;
constexpr int BLOCK_SIZE_NORMALS = 32;

__device__ __forceinline__ float2 operator+(const float2 a, const float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ void operator+=(float2 &a, const float2 b)
{
  a.x += b.x;
  a.y += b.y;
}

__device__ float evaluate(float2 z, const float2 c)
{
  constexpr int MAX_ITERS = 1000;
  constexpr float R = 2.0f;

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

void compute_julia_cuda(int width, int height, double c_re, double c_im, double zoomLevel,
                        double x_offset, double y_offset, float *buffer, cudaStream_t stream)
{
  float2 c = make_float2(c_re, c_im);
  float2 offsets = make_float2(x_offset, y_offset);

  unsigned int n_blocks = (width + BLOCK_SIZE_JULIA - 1) / BLOCK_SIZE_JULIA;
  dim3 block_dims{BLOCK_SIZE_JULIA, BLOCK_SIZE_JULIA};
  dim3 grid_dims{n_blocks, n_blocks};
  julia<<<grid_dims, block_dims, 0, stream>>>(buffer, (float)(1.0 / zoomLevel),
                                              offsets, c, width, height);
  CUDA_CHECK(cudaGetLastError());
}

template <int BLOCK_H, int BLOCK_W>
__global__ void compute_normals(const float *__restrict__ const h, float2 *__restrict__ out,
                                const int height, const int width)
{
  constexpr int HALO = 1;
  constexpr int TILE_H = BLOCK_H + 2 * HALO;
  constexpr int TILE_W = BLOCK_W + 2 * HALO;
  constexpr int NUM_THREADS = BLOCK_H * BLOCK_W;

  int x_idx = BLOCK_W * blockIdx.x + threadIdx.x;
  int y_idx = BLOCK_H * blockIdx.y + threadIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float s_h[TILE_H][TILE_W];

  // load h tile to smem
  int tid = threadIdx.y * BLOCK_W + threadIdx.x;
  int tile_top_left_x = blockIdx.x * BLOCK_W - HALO;
  int tile_top_left_y = blockIdx.y * BLOCK_H - HALO;

  for (int i = tid; i < TILE_H * TILE_W; i += NUM_THREADS)
  {
    int ly = i / TILE_W; // div and mod are optimized away because constexpr
    int lx = i % TILE_W;

    int gy = tile_top_left_y + ly;
    int gx = tile_top_left_x + lx;

    if (gx >= 0 && gx < width && gy >= 0 && gy < height)
      s_h[ly][lx] = __fsqrt_rz(__fsqrt_rz(h[gy * width + gx])) * 0.075f; // TODO: decide
    else
      s_h[ly][lx] = 0.0f;
  }
  __syncthreads();

  auto getUpperNormal = [&](int x, int y)
  {
    float h00 = s_h[y][x];
    float h10 = s_h[y][x + 1];
    float h11 = s_h[y + 1][x + 1];
    return float2{h00 - h10, h10 - h11};
  };

  auto getLowerNormal = [&](int x, int y)
  {
    float h00 = s_h[y][x];
    float h01 = s_h[y + 1][x];
    float h11 = s_h[y + 1][x + 1];
    return float2{h01 - h11, h00 - h01};
  };

  if (x_idx < width && y_idx < height)
  {
    // accumulate normals
    int sx = tx + HALO;
    int sy = ty + HALO;
    float2 myNormal = getUpperNormal(sx, sy);
    myNormal += getLowerNormal(sx, sy);
    myNormal += getUpperNormal(sx - 1, sy - 1);
    myNormal += getLowerNormal(sx - 1, sy - 1);
    myNormal += getUpperNormal(sx - 1, sy);
    myNormal += getLowerNormal(sx, sy - 1);

    out[2 * (y_idx * width + x_idx) + 1] = myNormal;
  }
}

// void compute_normals_cuda(const ProgramState &state, float *const h, float *out, cudaStream_t stream)
// {
//   unsigned int grid_h = (state.height + BLOCK_SIZE_NORMALS - 1) / BLOCK_SIZE_NORMALS;
//   unsigned int grid_w = (state.width + BLOCK_SIZE_NORMALS - 1) / BLOCK_SIZE_NORMALS;
//   dim3 grid_dims{grid_w, grid_h};
//   dim3 block_dims{BLOCK_SIZE_NORMALS, BLOCK_SIZE_NORMALS};

//   float2 *out_f2 = reinterpret_cast<float2 *>(out);
//   compute_normals<BLOCK_SIZE_NORMALS, BLOCK_SIZE_NORMALS>
//       <<<grid_dims, block_dims, 0, stream>>>(h, out_f2, state.height, state.width);
//   CUDA_CHECK(cudaGetLastError());
// }
