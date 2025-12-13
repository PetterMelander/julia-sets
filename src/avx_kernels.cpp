#include "avx_kernels.h"

#include <cmath>
#include <immintrin.h>
#include <omp.h>

constexpr int MAX_ITERS = 1000;
constexpr float R_s = 2.0f;
constexpr double R_d = 2.0;
constexpr int VEC_SIZE_SP = 16;
constexpr int VEC_SIZE_DP = 8;

__m512 evaluate(__m512 zReal, __m512 zImag, __m512 cReal, __m512 cImag)
{
  __m512 R2 = _mm512_set1_ps(R_s * R_s);
  __mmask16 active = 0xFFFF;
  __m512 escapeIter = _mm512_set1_ps(MAX_ITERS);
  __m512 escapeAbs2 = _mm512_set1_ps(0.0f);

  for (int i = 0; i < MAX_ITERS; ++i)
  {

    // compute squared magnitude of z
    __m512 zReal2 = _mm512_mul_ps(zReal, zReal);
    __m512 zImag2 = _mm512_mul_ps(zImag, zImag);
    __m512 zAbs2 = _mm512_add_ps(zReal2, zImag2);

    // compare with R^2 to see which active elements escaped
    __mmask16 escaped = _mm512_mask_cmp_ps_mask(active, zAbs2, R2, _CMP_GE_OQ);

    // save escape iteration and square magnitude
    __m512 iter = _mm512_set1_ps(i);
    escapeIter = _mm512_mask_blend_ps(escaped, escapeIter, iter);
    escapeAbs2 = _mm512_mask_blend_ps(escaped, escapeAbs2, zAbs2);

    // update active elements and break if done
    active = active & ~escaped;
    if (active == 0)
      break;

    // iterate z
    __m512 tmp = _mm512_mul_ps(_mm512_set1_ps(2.0f), zReal);
    zImag = _mm512_fmadd_ps(tmp, zImag, cImag);
    zReal = _mm512_add_ps(_mm512_sub_ps(zReal2, zImag2), cReal);
  }

  // postprocess to reduce color banding: iter + 1 - log(log(abs(z)))/log(2)
  __m512 smoothing = _mm512_sqrt_ps(escapeAbs2);

  // let compiler auto vectorize log function since it is not avx intrinsic
  alignas(64) float temp[VEC_SIZE_SP];
  _mm512_store_ps(temp, smoothing);
  for (int i = 0; i < VEC_SIZE_SP; ++i)
  {
    temp[i] = logf(logf(temp[i]));
  }
  smoothing = _mm512_load_ps(temp);

  __m512 negInvLog2 = _mm512_set1_ps(-1.0f / logf(2.0f));
  __m512 one = _mm512_set1_ps(1.0f);
  smoothing = _mm512_fmadd_ps(smoothing, negInvLog2, one);
  escapeIter = _mm512_mask_add_ps(escapeIter, ~active, escapeIter, smoothing);
  return escapeIter;
}

__m512d evaluate(__m512d zReal, __m512d zImag, __m512d cReal, __m512d cImag)
{
  __m512d R2 = _mm512_set1_pd(R_d * R_d);
  __mmask8 active = 0xFF;
  __m512d escapeIter = _mm512_set1_pd(MAX_ITERS);
  __m512d escapeAbs2 = _mm512_set1_pd(0.0);

  for (int i = 0; i < MAX_ITERS; ++i)
  {

    // compute squared magnitude of z
    __m512d zReal2 = _mm512_mul_pd(zReal, zReal);
    __m512d zImag2 = _mm512_mul_pd(zImag, zImag);
    __m512d zAbs2 = _mm512_add_pd(zReal2, zImag2);

    // compare with R^2 to see which active elements escaped
    __mmask8 escaped = _mm512_mask_cmp_pd_mask(active, zAbs2, R2, _CMP_GE_OQ);

    // save escape iteration and square magnitude
    __m512d iter = _mm512_set1_pd(i);
    escapeIter = _mm512_mask_blend_pd(escaped, escapeIter, iter);
    escapeAbs2 = _mm512_mask_blend_pd(escaped, escapeAbs2, zAbs2);

    // update active elements and break if done
    active = active & ~escaped;
    if (active == 0)
      break;

    // iterate z
    __m512d tmp = _mm512_mul_pd(_mm512_set1_pd(2.0), zReal);
    zImag = _mm512_fmadd_pd(tmp, zImag, cImag);
    zReal = _mm512_add_pd(_mm512_sub_pd(zReal2, zImag2), cReal);
  }

  // postprocess to reduce color banding: iter + 1 - log(log(abs(z)))/log(2)
  __m512d smoothing = _mm512_sqrt_pd(escapeAbs2);

  // let compiler auto vectorize log function since it is not avx intrinsic
  alignas(64) double temp[VEC_SIZE_DP];
  _mm512_store_pd(temp, smoothing);
  for (int i = 0; i < VEC_SIZE_DP; ++i)
  {
    temp[i] = log(log(temp[i]));
  }
  smoothing = _mm512_load_pd(temp);

  __m512d negInvLog2 = _mm512_set1_pd(-1.0 / log(2.0));
  __m512d one = _mm512_set1_pd(1.0);
  smoothing = _mm512_fmadd_pd(smoothing, negInvLog2, one);
  escapeIter = _mm512_mask_add_pd(escapeIter, ~active, escapeIter, smoothing);
  return escapeIter;
}

void julia(unsigned char *intensities, float range, float xOffset,
           float yOffset, float cReal, float cImag, int width, int height)
{

  // vectorize c
  __m512 cRealVec = _mm512_set1_ps(cReal);
  __m512 cImagVec = _mm512_set1_ps(cImag);

  // get deltas for vectorizing real part
  float reDelta = (2.0f * range) / (width - 1);
  __m512 reDeltaVec = _mm512_set1_ps(reDelta);
  __m512i indexIvec = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  __m512 indexVec = _mm512_cvtepi32_ps(indexIvec);

#pragma omp parallel for schedule(dynamic)
  for (int y = 0; y < height; ++y)
  {
    // vectorize imaginary part (const across vector)
    float im = ((float)y / (height - 1)) * range * 2 - range - yOffset;
    __m512 zImagVec = _mm512_set1_ps(im);

    for (int x = 0; x < width; x += VEC_SIZE_SP)
    {
      // vectorize real part (16 consecutive pixels in a row)
      float re = ((float)x / (width - 1)) * range * 2 - range - xOffset;
      __m512 zRealVec = _mm512_set1_ps(re);
      zRealVec = _mm512_fmadd_ps(indexVec, reDeltaVec, zRealVec);

      // evaluate pixels
      __m512 resultVec = evaluate(zRealVec, zImagVec, cRealVec, cImagVec);
      _mm512_store_ps(intensities + y * width + x, resultVec);
    }
  }
}

void julia(float *intensities, double range, double xOffset, double yOffset,
           double cReal, double cImag, int width, int height)
{

  // vectorize c
  __m512d cRealVec = _mm512_set1_pd(cReal);
  __m512d cImagVec = _mm512_set1_pd(cImag);

  // get deltas for vectorizing real part
  double reDelta = (2.0 * range) / (width - 1);
  __m512d reDeltaVec = _mm512_set1_pd(reDelta);
  __m512i indexIvec = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  __m512d indexVec = _mm512_cvtepi64_pd(indexIvec);

#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < height; ++y)
  {
    // vectorize imaginary part (const across vector)
    double im = ((double)y / (height - 1)) * range * 2 - range - yOffset;
    __m512d zImagVec = _mm512_set1_pd(im);

    for (int x = 0; x < width; x += VEC_SIZE_DP)
    {
      // vectorize real part (8 consecutive pixels in a row)
      double re = ((double)x / (width - 1)) * range * 2 - range - xOffset;
      __m512d zRealVec = _mm512_set1_pd(re);
      zRealVec = _mm512_fmadd_pd(indexVec, reDeltaVec, zRealVec);

      // evaluate pixels
      __m512d resultVec = evaluate(zRealVec, zImagVec, cRealVec, cImagVec);

      // convert to float and store
      __m256 float_vec = _mm512_cvtpd_ps(resultVec);
      _mm256_store_ps(intensities + y * width + x, float_vec);
    }
  }
}

void computeJuliaAvx(int width, int height, std::complex<double> c,
                     double zoomLevel, double xOffset, double yOffset,
                     float *buffer)
{
  julia(buffer, 1.0 / zoomLevel, xOffset, yOffset, c.real(), c.imag(), width, height);
}