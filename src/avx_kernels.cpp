#include "avx_kernels.h"

#include <cmath>
#include <immintrin.h>
#include <omp.h>

constexpr double R = 2.0;
constexpr int VEC_SIZE_PS = 16;
constexpr int VEC_SIZE_PD = 8;

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

inline bool allBlack(int maxIters, QuadResult res)
{
  return _mm512_cmp_ps_mask(res.data[0], _mm512_set1_ps((float)maxIters),
                            _CMP_EQ_OQ) == 0xFFFF &&
         _mm512_cmp_ps_mask(res.data[1], _mm512_set1_ps((float)maxIters),
                            _CMP_EQ_OQ) == 0xFFFF;
}

template <int maxIters, bool doSmoothing>
inline QuadResult evaluate(const __m512d zr[4], const __m512d zi[4],
                           const __m512d cReal, const __m512d cImag)
{
  const __m512d R2 = _mm512_set1_pd(R * R);
  const __m512d ones = _mm512_set1_pd(1.0);
  const __m512 twosPs = _mm512_set1_ps(2.0f);

  __mmask8 active[] = {0xFF, 0xFF, 0xFF, 0xFF};
  __m512d zReal[4] = {zr[0], zr[1], zr[2], zr[3]};
  __m512d zImag[4] = {zi[0], zi[1], zi[2], zi[3]};
  __m512d escapeIter[] = {_mm512_set1_pd(maxIters), _mm512_set1_pd(maxIters),
                          _mm512_set1_pd(maxIters), _mm512_set1_pd(maxIters)};
  __m512d escapeAbs2[4];

  __m512d curIter = _mm512_setzero_pd();
  for (int i = 0; i < maxIters; ++i)
  {

    // calc tmp component for zReal early to free register holding zImag2
#define STEP(k)                                                            \
  if (active[k])                                                           \
  {                                                                        \
    __m512d zImag2 = _mm512_mul_pd(zImag[k], zImag[k]);                    \
    __m512d tmpRe = _mm512_sub_pd(cReal, zImag2);                          \
                                                                           \
    __m512d zAbs2 = _mm512_fmadd_pd(zReal[k], zReal[k], zImag2);           \
    __mmask8 escaped =                                                     \
        _mm512_mask_cmp_pd_mask(active[k], zAbs2, R2, _CMP_GE_OQ);         \
                                                                           \
    escapeIter[k] = _mm512_mask_blend_pd(escaped, escapeIter[k], curIter); \
    if constexpr (doSmoothing)                                             \
      escapeAbs2[k] = _mm512_mask_blend_pd(escaped, escapeAbs2[k], zAbs2); \
                                                                           \
    active[k] = active[k] & ~escaped;                                      \
                                                                           \
    __m512d twoRe = _mm512_add_pd(zReal[k], zReal[k]);                     \
    zImag[k] = _mm512_fmadd_pd(twoRe, zImag[k], cImag);                    \
                                                                           \
    zReal[k] = _mm512_fmadd_pd(zReal[k], zReal[k], tmpRe);                 \
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
    alignas(64) float temp[VEC_SIZE_PS];
    _mm512_store_ps(temp, pdToPs(escapeAbs2[2 * i], escapeAbs2[2 * i + 1]));

#pragma omp simd
    for (int j = 0; j < VEC_SIZE_PS; ++j)
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
    result.data[i] =
        _mm512_mask_add_ps(escapeIterPs, ~mask, escapeIterPs, smoothing);
  }
  return result;
}

template <int maxIters>
void juliaKernelAvx(float *intensities,
                    double planeTopLeftX, double planeTopLeftY,
                    double pixelStepX, double pixelStepY,
                    double cReal, double cImag,
                    int width, int height)
{
  constexpr int BOX_SIZE = 32;

  const __m512d cRealVec = _mm512_set1_pd(cReal);
  const __m512d cImagVec = _mm512_set1_pd(cImag);
  const __m512d pixelStepVecX = _mm512_set1_pd(pixelStepX);
  const __m512d pixelStepVecY = _mm512_set1_pd(pixelStepY);

  const __m512d indexVec =
      _mm512_cvtepi64_pd(_mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0));
  const __m512d xOffsets[4] = {
      _mm512_set1_pd(0.0), _mm512_set1_pd(8.0 * pixelStepX),
      _mm512_set1_pd(16.0 * pixelStepX), _mm512_set1_pd(24.0 * pixelStepX)};

#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < height; y += BOX_SIZE)
  {
    for (int x = 0; x < width; x += BOX_SIZE)
    {
      __m512d zrBase[4], ziBase[4];
      double baseRe = planeTopLeftX + x * pixelStepX;
      __m512d baseReVec =
          _mm512_fmadd_pd(indexVec, pixelStepVecX, _mm512_set1_pd(baseRe));

      for (int k = 0; k < 4; ++k)
      {
        zrBase[k] = _mm512_add_pd(baseReVec, xOffsets[k]);
      }

      // process top row
      double topIm = planeTopLeftY + y * pixelStepY;
      for (int k = 0; k < 4; ++k)
        ziBase[k] = _mm512_set1_pd(topIm);

      QuadResult resTop = evaluate<maxIters, true>(zrBase, ziBase, cRealVec, cImagVec);

      // store top row
      for (int k = 0; k < 2; ++k)
      {
        _mm512_stream_ps(intensities + y * width + (x + k * VEC_SIZE_PS),
                         resTop.data[k]);
      }

      // process bottom row
      double botIm = planeTopLeftY + (y + BOX_SIZE - 1) * pixelStepY;
      for (int k = 0; k < 4; ++k)
        ziBase[k] = _mm512_set1_pd(botIm);

      QuadResult resBot = evaluate<maxIters, true>(zrBase, ziBase, cRealVec, cImagVec);

      // store bottom row
      for (int k = 0; k < 2; ++k)
      {
        _mm512_stream_ps(intensities + (y + BOX_SIZE - 1) * width + (x + k * VEC_SIZE_PS),
                         resBot.data[k]);
      }

      bool boxIsBlack = allBlack(maxIters, resTop) && allBlack(maxIters, resBot);

      // process sides (if top/bottom were black)
      if (boxIsBlack)
      {
        __m512d zrLeft[4], zrRight[4], ziVert[4];

        double baseX = planeTopLeftX + x * pixelStepX;
        double rightX = planeTopLeftX + (x + BOX_SIZE - 1) * pixelStepX;
        double baseY = planeTopLeftY + y * pixelStepY;

        __m512d baseYVec =
            _mm512_fmadd_pd(indexVec, pixelStepVecY, _mm512_set1_pd(baseY));

        for (int k = 0; k < 4; ++k)
        {
          zrLeft[k] = _mm512_set1_pd(baseX);
          zrRight[k] = _mm512_set1_pd(rightX);
          __m512d yOff = _mm512_set1_pd((k * 8) * pixelStepY);
          ziVert[k] = _mm512_add_pd(baseYVec, yOff);
        }

        // check left
        QuadResult resLeft =
            evaluate<maxIters, false>(zrLeft, ziVert, cRealVec, cImagVec);
        if (!allBlack(maxIters, resLeft))
          boxIsBlack = false;

        // check right (only if still black)
        if (boxIsBlack)
        {
          QuadResult resRight =
              evaluate<maxIters, false>(zrRight, ziVert, cRealVec, cImagVec);
          if (!allBlack(maxIters, resRight))
            boxIsBlack = false;
        }
      }

      if (boxIsBlack)
      {
        // fill interior
        __m512 blackVec = _mm512_set1_ps((float)maxIters);
        for (int k = 1; k < BOX_SIZE - 1; ++k)
        {
          for (int j = 0; j < BOX_SIZE / VEC_SIZE_PS; ++j)
          {
            _mm512_stream_ps(
                intensities + (y + k) * width + x + j * VEC_SIZE_PS, blackVec);
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

          QuadResult rowRes =
              evaluate<maxIters, true>(zrBase, ziBase, cRealVec, cImagVec);

          for (int v = 0; v < 2; ++v)
          {
            _mm512_stream_ps(intensities + (y + k) * width + x +
                                 v * VEC_SIZE_PS,
                             rowRes.data[v]);
          }
        }
      }
    }
  }
}

void computeJuliaAvx(int width, int height, bool screenGrab, std::complex<double> c,
                     double zoomLevel, double xOffset, double yOffset, float *buffer)
{

  int minDim = std::min(width, height);
  double viewWidth = 2.0 * ((double)width / (minDim * zoomLevel));
  double viewHeight = 2.0 * ((double)height / (minDim * zoomLevel));

  double pixelStepX = viewWidth / (width - 1);
  double pixelStepY = viewHeight / (height - 1);

  double planeTopLeftX = -(viewWidth / 2.0) - (double)xOffset;
  double planeTopLeftY = -(viewHeight / 2.0) - (double)yOffset;

  if (screenGrab)
    juliaKernelAvx<1000000>(buffer, planeTopLeftX, planeTopLeftY, pixelStepX, pixelStepY,
                            c.real(), c.imag(), width, height);
  else
    juliaKernelAvx<2500>(buffer, planeTopLeftX, planeTopLeftY, pixelStepX, pixelStepY,
                         c.real(), c.imag(), width, height);
}