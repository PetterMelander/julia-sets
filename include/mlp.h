#pragma once

#include <cmath>
#include <immintrin.h>

#include "mlp_constants.h"

inline void mlpLayer(
    const float *__restrict__ inBuf,
    float *__restrict__ outBuf,
    const float *__restrict__ wBuf,
    const float *__restrict__ bBuf,
    const int numInputs,
    const int numOutputs)
{
    // loop over input nodes, vectorize associated weights, and accumulate to output nodes
    for (int i = 0; i < numInputs; ++i)
    {
        __m512 input = _mm512_set1_ps(inBuf[i]);
        for (int j = 0; j < numOutputs; j += 16)
        {
            __m512 weights = _mm512_load_ps(wBuf + i * numOutputs + j);
            __m512 accum = _mm512_load_ps(outBuf + j);
            accum = _mm512_fmadd_ps(weights, input, accum);

            if (i == numInputs - 1) // final iteration
            {
                // bias
                __m512 biases = _mm512_load_ps(bBuf + j);
                accum = _mm512_add_ps(accum, biases);

                // relu
                __m512 zeros = _mm512_set1_ps(0.0f);
                accum = _mm512_max_ps(accum, zeros);
            }
            _mm512_store_ps(outBuf + j, accum);
        }
    }
}

inline float outputLayer(
    const float *__restrict__ inBuf,
    const float *__restrict__ wBuf,
    const float b,
    const int numInputs)
{
    float accum = 0.0f;
    for (int i = 0; i < numInputs; i += 16)
    {
        __m512 weights = _mm512_load_ps(wBuf + i);
        __m512 inputs = _mm512_load_ps(inBuf + i);
        __m512 nodeContribs = _mm512_mul_ps(weights, inputs);
        accum += _mm512_reduce_add_ps(nodeContribs);
    }
    return accum + b;
}

inline float mlpPredict(const float *scaledInputs)
{
    alignas(64) float buffer1[LAYER_SIZES[0]] = {};
    alignas(64) float buffer2[LAYER_SIZES[1]] = {};
    mlpLayer(scaledInputs, buffer1, W0, B0, N_INPUTS, LAYER_SIZES[0]);
    mlpLayer(buffer1, buffer2, W1, B1, LAYER_SIZES[0], LAYER_SIZES[1]);
    return outputLayer(buffer2, W2, B2[0], LAYER_SIZES[1]);
}
