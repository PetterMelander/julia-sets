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
  constexpr int MAX_ITERS = 2500;
  constexpr float R = 2.0f;

  float escapeIter = MAX_ITERS;
  float escapeAbs2 = 0.0f;
  for (int i = 0; i < MAX_ITERS; ++i)
  {
    float abs2 = z.x * z.x + z.y * z.y;
    if (abs2 >= R * R)
    {
      escapeIter = i;
      escapeAbs2 = abs2;
      break;
    }
    float re = z.x * z.x - z.y * z.y + c.x;
    float im = 2.0f * z.x * z.y + c.y;
    z.x = re;
    z.y = im;
  }
  float retval;
  if (escapeIter < MAX_ITERS)
  {
    // 1 - log(log(abs(z)))/log(2) = 2 - log2(ln(abs(z²)))
    retval = escapeIter + 2.0f - __log2f(__logf(escapeAbs2));
  }
  else
  {
    retval = MAX_ITERS;
  }
  return retval;
}

__global__ void julia(float *const __restrict__ buffer, const float2 planeTopLeft,
                      const float2 pixelStep, const float2 c, const int width, const int height)
{
  int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

  if (xIdx < width && yIdx < height)
  {
    float re = planeTopLeft.x + (float)xIdx * pixelStep.x;
    float im = planeTopLeft.y + (float)yIdx * pixelStep.y;
    float2 z{re, im};

    float intensity = evaluate(z, c);
    buffer[yIdx * width + xIdx] = intensity;
  }
}

void computeJuliaCuda(int width, int height, std::complex<double> c, double zoomLevel,
                      double xOffset, double yOffset, float *buffer, cudaStream_t stream)
{
  float2 C = make_float2(c.real(), c.imag());

  int minDim = std::min(width, height);
  float viewWidth = 2.0f * ((float)width / (minDim * zoomLevel));
  float viewHeight = 2.0f * ((float)height / (minDim * zoomLevel));
  
  float2 pixelStep;
  pixelStep.x = viewWidth / (width - 1);
  pixelStep.y = viewHeight / (height - 1);

  float2 planeTopLeft;
  planeTopLeft.x = -(viewWidth / 2.0f) - (float)xOffset;
  planeTopLeft.y = -(viewHeight / 2.0f) - (float)yOffset;

  unsigned int gridHeight = (height + BLOCK_SIZE_JULIA - 1) / BLOCK_SIZE_JULIA;
  unsigned int gridWidth = (width + BLOCK_SIZE_JULIA - 1) / BLOCK_SIZE_JULIA;
  dim3 gridDims{gridWidth, gridHeight};
  dim3 blockDims{BLOCK_SIZE_JULIA, BLOCK_SIZE_JULIA};
  julia<<<gridDims, blockDims, 0, stream>>>(buffer, planeTopLeft, pixelStep, C, width, height);
  CUDA_CHECK(cudaGetLastError());
}

template <int BLOCK_H, int BLOCK_W>
__global__ void compute_normals(const float *__restrict__ const h, float2 *__restrict__ out,
                                const int height, const int width)
{
  /*
  This kernel requires some explanation. It computes the normal for each vertex as the area weighted
  average of all 6 triangles it belongs to.

  Since the x and z values (here called x and y) are a constant grid, the normals really only
  depend on the y values (here called h). If xstep is the difference in x between two horizontally
  adjacent vertices and ystep is the difference in y between two vertically adjacent vertices, then
  the normal for a TOP triangle (upper right half of a rectangle) is the following:

  (
    ystep * (h00 - h10),
    xstep * ystep,
    xstep * (h10 - h11)
  )

  and for a BOTTOM triangle:

  (
    ystep * (h01 - h11),
    xstep * ystep,
    xstep * (h00 - h01)
  ),

  where h00 is the top left corner of the triangle, and in clockwise direction, the remaining
  corners are h10, h11, and h01.

  To reduce memory bandwidth usage, the constant y coordinate is not written to global memory. It is
  instead left to the vertex shader to compute. Similarly, the constants ystep and xstep from the
  x and z coordinates are left to the vertex shader to compute.
  */
  constexpr int HALO = 1;
  constexpr int TILE_H = BLOCK_H + 2 * HALO;
  constexpr int TILE_W = BLOCK_W + 2 * HALO;
  constexpr int NUM_THREADS = BLOCK_H * BLOCK_W;

  int xIdx = BLOCK_W * blockIdx.x + threadIdx.x;
  int yIdx = BLOCK_H * blockIdx.y + threadIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float s_h[TILE_H][TILE_W];

  // load h tile to smem
  int tid = threadIdx.y * BLOCK_W + threadIdx.x;
  int tileTopLeftX = blockIdx.x * BLOCK_W - HALO;
  int tileTopLeftY = blockIdx.y * BLOCK_H - HALO;

  for (int i = tid; i < TILE_H * TILE_W; i += NUM_THREADS)
  {
    int ly = i / TILE_W; // div and mod are optimized away because constexpr
    int lx = i % TILE_W;

    int gy = tileTopLeftY + ly;
    int gx = tileTopLeftX + lx;

    if (gx >= 0 && gx < width && gy >= 0 && gy < height)
      s_h[ly][lx] = h[gy * width + gx];
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

  if (xIdx < width && yIdx < height)
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

    out[2 * (yIdx * width + xIdx) + 1] = myNormal;
  }
}

void computeNormalsCuda(int width, int height, float *const h, float *out, cudaStream_t stream)
{
  unsigned int gridHeight = (height + BLOCK_SIZE_NORMALS - 1) / BLOCK_SIZE_NORMALS;
  unsigned int gridWidth = (width + BLOCK_SIZE_NORMALS - 1) / BLOCK_SIZE_NORMALS;
  dim3 gridDims{gridWidth, gridHeight};
  dim3 blockDims{BLOCK_SIZE_NORMALS, BLOCK_SIZE_NORMALS};

  float2 *out_f2 = reinterpret_cast<float2 *>(out);
  compute_normals<BLOCK_SIZE_NORMALS, BLOCK_SIZE_NORMALS>
      <<<gridDims, blockDims, 0, stream>>>(h, out_f2, height, width);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void scaleImage(const int dsize, const float *__restrict__ const imgMin,
                           const float *__restrict__ const imgMax, float *__restrict__ h)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ float scale;
  __shared__ float min;
  if (threadIdx.x == 0)
  {
    min = __ldg(imgMin);
    float max = __ldg(imgMax);
    scale = 1.0f / (max - min);
    // scale = 5.0f / (max - min);
  }
  __syncthreads();
  if (idx < dsize)
  {
    h[idx] = (h[idx] - min) * scale * 0.25f;
    // h[idx] = expf(-(h[idx] - min) * scale) * 0.25f;
  }
}

void rescaleImage(int width, int height, float *imgMin, float *imgMax, float *h, cudaStream_t stream)
{
  unsigned int dsize = width * height;
  int block_size = 512;
  int num_blocks = (dsize + block_size - 1) / block_size;
  scaleImage<<<num_blocks, block_size, 0, stream>>>(dsize, imgMin, imgMax, h);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void updateScale(float *__restrict__ const oldMin, float *__restrict__ const newMin, float *__restrict__ const oldMax, float *__restrict__ const newMax)
{
  if (threadIdx.x == 0)
  {
    *newMin = 0.99f * *oldMin + 0.01f * *newMin;
    *newMax = 0.99f * *oldMax + 0.01f * *newMax;
    *oldMin = *newMin;
    *oldMax = *newMax;
  }
}

void updateScale(float *oldMin, float *newMin, float *oldMax, float *newMax, cudaStream_t stream)
{
  updateScale<<<1, 1, 0, stream>>>(oldMin, newMin, oldMax, newMax);
}
