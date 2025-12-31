#include "avx_kernels.h"

#include <cmath>
#include <immintrin.h>
#include <omp.h>

constexpr int MAX_ITERS = 2500;
constexpr float R_s = 2.0f;
constexpr double R_d = 2.0;
constexpr int VEC_SIZE_SP = 16;
constexpr int VEC_SIZE_DP = 8;

inline __m512 pdToPs(const __m512d part1, const __m512d part2)
{
  __m256 lower = _mm512_cvtpd_ps(part1);
  __m256 upper = _mm512_cvtpd_ps(part2);

  __m512 vec_lower = _mm512_castps256_ps512(lower);
  return _mm512_insertf32x8(vec_lower, upper, 1);
}

struct QuadResult
{
  __m512 data[2];
};

inline bool allBlack(QuadResult res)
{
  return _mm512_cmp_ps_mask(res.data[0], _mm512_set1_ps((float)MAX_ITERS), _CMP_EQ_OQ) == 0xFFFF && _mm512_cmp_ps_mask(res.data[1], _mm512_set1_ps((float)MAX_ITERS), _CMP_EQ_OQ) == 0xFFFF;
}

template <bool doSmoothing>
inline QuadResult evaluate(
    const __m512d zr[4],
    const __m512d zi[4],
    const __m512d cReal,
    const __m512d cImag)
{
  const __m512d R2 = _mm512_set1_pd(R_d * R_d);
  const __m512d ones = _mm512_set1_pd(1.0);
  const __m512d twosPd = _mm512_set1_pd(2.0);
  const __m512 twosPs = _mm512_set1_ps(2.0f);

  __mmask8 active[] = {0xFF, 0xFF, 0xFF, 0xFF};
  __m512d zReal[4] = {zr[0], zr[1], zr[2], zr[3]};
  __m512d zImag[4] = {zi[0], zi[1], zi[2], zi[3]};
  __m512d escapeIter[] = {
      _mm512_set1_pd(MAX_ITERS),
      _mm512_set1_pd(MAX_ITERS),
      _mm512_set1_pd(MAX_ITERS),
      _mm512_set1_pd(MAX_ITERS)};
  __m512d escapeAbs2[4];

  __m512d curIter = _mm512_setzero_pd();
  for (int i = 0; i < MAX_ITERS; ++i)
  {

#define STEP(k)                                                                   \
  if (active[k])                                                                  \
  {                                                                               \
    __m512d zReal2 = _mm512_mul_pd(zReal[k], zReal[k]);                           \
    __m512d zImag2 = _mm512_mul_pd(zImag[k], zImag[k]);                           \
    __m512d zAbs2 = _mm512_add_pd(zReal2, zImag2);                                \
                                                                                  \
    __mmask8 escaped = _mm512_mask_cmp_pd_mask(active[k], zAbs2, R2, _CMP_GE_OQ); \
                                                                                  \
    escapeIter[k] = _mm512_mask_blend_pd(escaped, escapeIter[k], curIter);        \
    if constexpr (doSmoothing)                                                    \
      escapeAbs2[k] = _mm512_mask_blend_pd(escaped, escapeAbs2[k], zAbs2);        \
                                                                                  \
    active[k] = active[k] & ~escaped;                                             \
                                                                                  \
    __m512d tmp = _mm512_mul_pd(twosPd, zReal[k]);                                \
    zImag[k] = _mm512_fmadd_pd(tmp, zImag[k], cImag);                             \
    zReal[k] = _mm512_add_pd(_mm512_sub_pd(zReal2, zImag2), cReal);               \
  }

    STEP(0);
    STEP(1);
    STEP(2);
    STEP(3);

#undef STEP

    if (!(active[0] | active[1] | active[2] | active[3]))
      break;

    curIter = _mm512_add_pd(curIter, ones);
  }

  QuadResult result;

  if constexpr (!doSmoothing)
  {
    result.data[0] = pdToPs(escapeIter[0], escapeIter[1]);
    result.data[1] = pdToPs(escapeIter[2], escapeIter[3]);
    return result;
  }

  for (int i = 0; i < 2; ++i)
  {
    alignas(64) float temp[VEC_SIZE_SP];
    _mm512_store_ps(temp, pdToPs(escapeAbs2[2 * i], escapeAbs2[2 * i + 1]));

#pragma omp simd
    for (int j = 0; j < VEC_SIZE_SP; ++j)
    {
      if (temp[j] > 1.0f)
        temp[j] = log2f(logf(temp[j]));
      else
        temp[j] = 0.0f;
    }
    __m512 smoothing = _mm512_load_ps(temp);
    smoothing = _mm512_sub_ps(twosPs, smoothing);
    __mmask16 mask = (active[2 * i]) | (active[2 * i + 1] << 8);

    __m512 escapeIterPs = pdToPs(escapeIter[2 * i], escapeIter[2 * i + 1]);
    result.data[i] = _mm512_mask_add_ps(escapeIterPs, ~mask, escapeIterPs, smoothing);
  }
  return result;
}

__m512 evaluate(__m512 zReal, __m512 zImag, __m512 cReal, __m512 cImag)
{
  const __m512 R2 = _mm512_set1_ps(R_s * R_s);
  __mmask16 active = 0xFFFF;
  __m512 escapeIter = _mm512_set1_ps(MAX_ITERS);
  __m512 escapeAbs2 = _mm512_set1_ps(0.0f);
  const __m512 twos = _mm512_set1_ps(2.0f);

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
    __m512 tmp = _mm512_mul_ps(twos, zReal);
    zImag = _mm512_fmadd_ps(tmp, zImag, cImag);
    zReal = _mm512_add_ps(_mm512_sub_ps(zReal2, zImag2), cReal);
  }

  // postprocess to reduce color banding: iter + 1 - log(log(abs(z)))/log(2) = iter + 2 - log2(ln(abs(z²)))
  // let compiler auto vectorize log function since it is not avx intrinsic
  alignas(64) float temp[VEC_SIZE_SP];
  _mm512_store_ps(temp, escapeAbs2);
  for (int i = 0; i < VEC_SIZE_SP; ++i)
  {
    temp[i] = log2f(logf(temp[i]));
  }
  __m512 smoothing = _mm512_load_ps(temp);
  smoothing = _mm512_sub_ps(twos, smoothing);
  escapeIter = _mm512_mask_add_ps(escapeIter, ~active, escapeIter, smoothing);
  return escapeIter;
}

__m512d evaluate(__m512d zReal, __m512d zImag, __m512d cReal, __m512d cImag)
{
  const __m512d R2 = _mm512_set1_pd(R_d * R_d);
  __mmask8 active = 0xFF;
  __m512d escapeIter = _mm512_set1_pd(MAX_ITERS);
  __m512d escapeAbs2 = _mm512_set1_pd(0.0);
  const __m512d twos = _mm512_set1_pd(2.0);

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
    __m512d tmp = _mm512_mul_pd(twos, zReal);
    zImag = _mm512_fmadd_pd(tmp, zImag, cImag);
    zReal = _mm512_add_pd(_mm512_sub_pd(zReal2, zImag2), cReal);
  }

  // postprocess to reduce color banding: iter + 1 - log(log(abs(z)))/log(2) = iter + 2 - log2(ln(abs(z²)))
  // let compiler auto vectorize log function since it is not avx intrinsic
  alignas(64) double temp[VEC_SIZE_DP];
  _mm512_store_pd(temp, escapeAbs2);
  for (int i = 0; i < VEC_SIZE_DP; ++i)
  {
    temp[i] = log2(log(temp[i]));
  }
  __m512d smoothing = _mm512_load_pd(temp);
  smoothing = _mm512_sub_pd(twos, smoothing);
  escapeIter = _mm512_mask_add_pd(escapeIter, ~active, escapeIter, smoothing);
  return escapeIter;
}

void juliaKernelAvx(float *intensities, float planeTopLeftX, float planeTopLeftY, float pixelStepX,
                    float pixelStepY, float cReal, double cImag, int width, int height)
{

  // vectorize c
  __m512 cRealVec = _mm512_set1_ps(cReal);
  __m512 cImagVec = _mm512_set1_ps(cImag);

  // get deltas for vectorizing real part
  __m512 pixelStepVecX = _mm512_set1_ps(pixelStepX);
  __m512i indexIvec = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  __m512 indexVec = _mm512_cvtepi32_ps(indexIvec);

#pragma omp parallel for schedule(dynamic)
  for (int y = 0; y < height; ++y)
  {
    // vectorize imaginary part (const across vector)
    float im = planeTopLeftY + y * pixelStepY;
    __m512 zImagVec = _mm512_set1_ps(im);

    for (int x = 0; x < width; x += VEC_SIZE_SP)
    {
      // vectorize real part (16 consecutive pixels in a row)
      float re = planeTopLeftX + x * pixelStepX;
      __m512 zRealVec = _mm512_set1_ps(re);
      zRealVec = _mm512_fmadd_ps(indexVec, pixelStepVecX, zRealVec);

      // evaluate pixels
      __m512 resultVec = evaluate(zRealVec, zImagVec, cRealVec, cImagVec);
      _mm512_store_ps(intensities + y * width + x, resultVec);
    }
  }
}

void juliaKernelAvx(
    float *intensities, double planeTopLeftX, double planeTopLeftY,
    double pixelStepX, double pixelStepY, double cReal, double cImag, int width, int height)
{
  constexpr int BOX_SIZE = 32;

  const __m512d cRealVec = _mm512_set1_pd(cReal);
  const __m512d cImagVec = _mm512_set1_pd(cImag);
  const __m512d pixelStepVecX = _mm512_set1_pd(pixelStepX);
  const __m512d pixelStepVecY = _mm512_set1_pd(pixelStepY);

  const __m512d indexVec = _mm512_cvtepi64_pd(_mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0));
  const __m512d xOffsets[4] = {
      _mm512_set1_pd(0.0),
      _mm512_set1_pd(8.0 * pixelStepX),
      _mm512_set1_pd(16.0 * pixelStepX),
      _mm512_set1_pd(24.0 * pixelStepX)};

#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < height; y += BOX_SIZE)
  {
    for (int x = 0; x < width; x += BOX_SIZE)
    {
      __m512d zrBase[4], ziBase[4];
      double baseRe = planeTopLeftX + x * pixelStepX;
      __m512d baseReVec = _mm512_fmadd_pd(indexVec, pixelStepVecX, _mm512_set1_pd(baseRe));

      for (int k = 0; k < 4; ++k)
      {
        zrBase[k] = _mm512_add_pd(baseReVec, xOffsets[k]);
      }

      // process top row
      double topIm = planeTopLeftY + y * pixelStepY;
      for (int k = 0; k < 4; ++k)
        ziBase[k] = _mm512_set1_pd(topIm);

      QuadResult resTop = evaluate<true>(zrBase, ziBase, cRealVec, cImagVec);

      // store top row
      for (int k = 0; k < 2; ++k)
      {
        _mm512_stream_ps(intensities + y * width + (x + k * VEC_SIZE_SP), resTop.data[k]);
      }

      // process bottom row
      double botIm = planeTopLeftY + (y + BOX_SIZE - 1) * pixelStepY;
      for (int k = 0; k < 4; ++k)
        ziBase[k] = _mm512_set1_pd(botIm);

      QuadResult resBot = evaluate<true>(zrBase, ziBase, cRealVec, cImagVec);

      // store bottom row
      for (int k = 0; k < 2; ++k)
      {
        _mm512_stream_ps(intensities + (y + BOX_SIZE - 1) * width + (x + k * VEC_SIZE_SP), resBot.data[k]);
      }

      bool boxIsBlack = allBlack(resTop) && allBlack(resBot);

      // process sides (if top/bottom were black)
      if (boxIsBlack)
      {
        __m512d zrLeft[4], zrRight[4], ziVert[4];

        double baseX = planeTopLeftX + x * pixelStepX;
        double rightX = planeTopLeftX + (x + BOX_SIZE - 1) * pixelStepX;
        double baseY = planeTopLeftY + y * pixelStepY;

        __m512d baseYVec = _mm512_fmadd_pd(indexVec, pixelStepVecY, _mm512_set1_pd(baseY));

        for (int k = 0; k < 4; ++k)
        {
          zrLeft[k] = _mm512_set1_pd(baseX);
          zrRight[k] = _mm512_set1_pd(rightX);
          __m512d yOff = _mm512_set1_pd((k * 8) * pixelStepY);
          ziVert[k] = _mm512_add_pd(baseYVec, yOff);
        }

        // check left
        QuadResult resLeft = evaluate<false>(zrLeft, ziVert, cRealVec, cImagVec);
        if (!allBlack(resLeft))
          boxIsBlack = false;

        // check right (only if still black)
        if (boxIsBlack)
        {
          QuadResult resRight = evaluate<false>(zrRight, ziVert, cRealVec, cImagVec);
          if (!allBlack(resRight))
            boxIsBlack = false;
        }
      }

      // fill interior
      if (boxIsBlack)
      {
        __m512 blackVec = _mm512_set1_ps((float)MAX_ITERS);
        for (int k = 1; k < BOX_SIZE - 1; ++k)
        {
          for (int j = 0; j < BOX_SIZE / VEC_SIZE_SP; ++j)
          {
            _mm512_stream_ps(intensities + (y + k) * width + x + j * VEC_SIZE_SP, blackVec);
          }
        }
      }
      else
      {
        // compute interior manually
        for (int k = 1; k < BOX_SIZE - 1; ++k)
        {
          double im = planeTopLeftY + (y + k) * pixelStepY;
          for (int v = 0; v < 4; ++v)
            ziBase[v] = _mm512_set1_pd(im);

          QuadResult rowRes = evaluate<true>(zrBase, ziBase, cRealVec, cImagVec);

          for (int v = 0; v < 2; ++v)
          {
            _mm512_stream_ps(intensities + (y + k) * width + (x + v * VEC_SIZE_SP), rowRes.data[v]);
          }
        }
      }
    }
  }
}

void computeJuliaAvx(int width, int height, std::complex<double> c, double zoomLevel,
                     double xOffset, double yOffset, float *buffer)
{

  int minDim = std::min(width, height);
  double viewWidth = 2.0 * ((double)width / (minDim * zoomLevel));
  double viewHeight = 2.0 * ((double)height / (minDim * zoomLevel));

  double pixelStepX = viewWidth / (width - 1);
  double pixelStepY = viewHeight / (height - 1);

  double planeTopLeftX = -(viewWidth / 2.0) - (double)xOffset;
  double planeTopLeftY = -(viewHeight / 2.0) - (double)yOffset;

  juliaKernelAvx(buffer, planeTopLeftX, planeTopLeftY, pixelStepX,
                 pixelStepY, c.real(), c.imag(), width, height);
}