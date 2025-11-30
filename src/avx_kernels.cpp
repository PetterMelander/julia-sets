#include "avx_kernels.h"
#include "gl_utils.h"

#include <cmath>
#include <immintrin.h>
#include <omp.h>

constexpr int MAX_ITERS = 1000;
constexpr float R_s = 2.0f;
constexpr double R_d = 2.0;
constexpr int VEC_SIZE_SP = 16;
constexpr int VEC_SIZE_DP = 8;

__m512 evaluate(__m512 z_re, __m512 z_im, __m512 c_re, __m512 c_im)
{
  __m512 R2 = _mm512_set1_ps(R_s * R_s);
  __mmask16 active = 0xFFFF;
  __m512 escape_iter = _mm512_set1_ps(MAX_ITERS);
  __m512 escape_abs2 = _mm512_set1_ps(0);

  for (int i = 0; i < MAX_ITERS; ++i)
  {

    // compute squared magnitude of z
    __m512 z_re2 = _mm512_mul_ps(z_re, z_re);
    __m512 z_im2 = _mm512_mul_ps(z_im, z_im);
    __m512 z_abs2 = _mm512_add_ps(z_re2, z_im2);

    // compare with R^2 to see which active elements escaped
    __mmask16 escaped = _mm512_mask_cmp_ps_mask(active, z_abs2, R2, _CMP_GE_OQ);

    // save escape iteration and square magnitude
    __m512 iter = _mm512_set1_ps(i);
    escape_iter = _mm512_mask_blend_ps(escaped, escape_iter, iter);
    escape_abs2 = _mm512_mask_blend_ps(escaped, escape_abs2, z_abs2);

    // update active elements and break if done
    active = active & ~escaped;
    if (active == 0)
      break;

    // iterate z
    __m512 tmp = _mm512_mul_ps(_mm512_set1_ps(2.f), z_re);
    z_im = _mm512_fmadd_ps(tmp, z_im, c_im);
    z_re = _mm512_add_ps(_mm512_sub_ps(z_re2, z_im2), c_re);
  }

  // postprocess to reduce color banding: iter + 1 - log(log(abs(z)))/log(2)
  __m512 smoothing = _mm512_sqrt_ps(escape_abs2);

  // let compiler auto vectorize log function since it is not avx intrinsic
  alignas(64) float temp[VEC_SIZE_SP];
  _mm512_store_ps(temp, smoothing);
  for (int i = 0; i < VEC_SIZE_SP; ++i)
  {
    temp[i] = logf(logf(temp[i]));
  }
  smoothing = _mm512_load_ps(temp);

  __m512 neg_inv_log2 = _mm512_set1_ps(-1.f / logf(2));
  __m512 one = _mm512_set1_ps(1.f);
  smoothing = _mm512_fmadd_ps(smoothing, neg_inv_log2, one);
  escape_iter = _mm512_mask_add_ps(escape_iter, ~active, escape_iter, smoothing);
  return escape_iter;
}

__m512d evaluate(__m512d z_re, __m512d z_im, __m512d c_re, __m512d c_im)
{
  __m512d R2 = _mm512_set1_pd(R_d * R_d);
  __mmask8 active = 0xFF;
  __m512d escape_iter = _mm512_set1_pd(MAX_ITERS);
  __m512d escape_abs2 = _mm512_set1_pd(0);

  for (int i = 0; i < MAX_ITERS; ++i)
  {

    // compute squared magnitude of z
    __m512d z_re2 = _mm512_mul_pd(z_re, z_re);
    __m512d z_im2 = _mm512_mul_pd(z_im, z_im);
    __m512d z_abs2 = _mm512_add_pd(z_re2, z_im2);

    // compare with R^2 to see which active elements escaped
    __mmask8 escaped = _mm512_mask_cmp_pd_mask(active, z_abs2, R2, _CMP_GE_OQ);

    // save escape iteration and square magnitude
    __m512d iter = _mm512_set1_pd(i);
    escape_iter = _mm512_mask_blend_pd(escaped, escape_iter, iter);
    escape_abs2 = _mm512_mask_blend_pd(escaped, escape_abs2, z_abs2);

    // update active elements and break if done
    active = active & ~escaped;
    if (active == 0)
      break;

    // iterate z
    __m512d tmp = _mm512_mul_pd(_mm512_set1_pd(2.0), z_re);
    z_im = _mm512_fmadd_pd(tmp, z_im, c_im);
    z_re = _mm512_add_pd(_mm512_sub_pd(z_re2, z_im2), c_re);
  }

  // postprocess to reduce color banding: iter + 1 - log(log(abs(z)))/log(2)
  __m512d smoothing = _mm512_sqrt_pd(escape_abs2);

  // let compiler auto vectorize log function since it is not avx intrinsic
  alignas(64) double temp[VEC_SIZE_DP];
  _mm512_store_pd(temp, smoothing);
  for (int i = 0; i < VEC_SIZE_DP; ++i)
  {
    temp[i] = log(log(temp[i]));
  }
  smoothing = _mm512_load_pd(temp);

  __m512d neg_inv_log2 = _mm512_set1_pd(-1.0 / log(2.0));
  __m512d one = _mm512_set1_pd(1.0);
  smoothing = _mm512_fmadd_pd(smoothing, neg_inv_log2, one);
  escape_iter = _mm512_mask_add_pd(escape_iter, ~active, escape_iter, smoothing);
  return escape_iter;
}

void julia(unsigned char *intensities, float range, float x_offset, float y_offset,
           float c_re, float c_im, int width, int height)
{

  // vectorize c
  __m512 c_re_vec = _mm512_set1_ps(c_re);
  __m512 c_im_vec = _mm512_set1_ps(c_im);

  // get deltas for vectorizing real part
  float re_delta = (2.f * range) / (width - 1);
  __m512 re_delta_vec = _mm512_set1_ps(re_delta);
  __m512i index_ivec =
      _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  __m512 index_vec = _mm512_cvtepi32_ps(index_ivec);

#pragma omp parallel for schedule(dynamic)
  for (int y = 0; y < height; ++y)
  {
    // vectorize imaginary part (const across vector)
    float im = ((float)y / (height - 1)) * range * 2 - range - y_offset;
    __m512 z_im_vec = _mm512_set1_ps(im);

    for (int x = 0; x < width; x += VEC_SIZE_SP)
    {
      // vectorize real part (16 consecutive pixels in a row)
      float re = ((float)x / (width - 1)) * range * 2 - range - x_offset;
      __m512 z_re_vec = _mm512_set1_ps(re);
      z_re_vec = _mm512_fmadd_ps(index_vec, re_delta_vec, z_re_vec);

      // evaluate pixels
      __m512 result_vec = evaluate(z_re_vec, z_im_vec, c_re_vec, c_im_vec);
      _mm512_store_ps(intensities + y * width + x, result_vec);
    }
  }
}

void julia(float *intensities, double range, double x_offset, double y_offset,
           double c_re, double c_im, int width, int height)
{

  // vectorize c
  __m512d c_re_vec = _mm512_set1_pd(c_re);
  __m512d c_im_vec = _mm512_set1_pd(c_im);

  // get deltas for vectorizing real part
  double re_delta = (2.0 * range) / (width - 1);
  __m512d re_delta_vec = _mm512_set1_pd(re_delta);
  __m512i index_ivec = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  __m512d index_vec = _mm512_cvtepi64_pd(index_ivec);

#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < height; ++y)
  {
    // vectorize imaginary part (const across vector)
    double im = ((double)y / (height - 1)) * range * 2 - range - y_offset;
    __m512d z_im_vec = _mm512_set1_pd(im);

    for (int x = 0; x < width; x += VEC_SIZE_DP)
    {
      // vectorize real part (8 consecutive pixels in a row)
      double re = ((double)x / (width - 1)) * range * 2 - range - x_offset;
      __m512d z_re_vec = _mm512_set1_pd(re);
      z_re_vec = _mm512_fmadd_pd(index_vec, re_delta_vec, z_re_vec);

      // evaluate pixels
      __m512d result_vec = evaluate(z_re_vec, z_im_vec, c_re_vec, c_im_vec);

      // convert to float and store
      __m256 float_vec = _mm512_cvtpd_ps(result_vec);
      _mm256_store_ps(intensities + y * width + x, float_vec);
    }
  }
}

void compute_julia_avx(ProgramState state, float *buffer)
{
  julia(buffer, 1.0 / state.zoomLevel, state.x_offset, state.y_offset,
        state.c_re, state.c_im, state.width, state.height);
}