#pragma once

#include <cmath>
#include <immintrin.h>

#include "mlp_constants.h"

void mlpLayer(const float *inBuf, float *outBuf, const float *wBuf,
              const float *bBuf, const int numInputs, const int numOutputs)
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

float outputLayer(const float *inBuf, const float *wBuf, const float b, const int numInputs)
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

float mlpPredict(const double *inputs)
{
    // scale inputs
    float scaled_inputs[] = {
        (float)(inputs[0] - INPUT_MEANS[0]) / INPUT_STDS[0],
        (float)(inputs[1] - INPUT_MEANS[1]) / INPUT_STDS[1],
        (float)(inputs[2] - INPUT_MEANS[2]) / INPUT_STDS[2],
        (float)(inputs[3] - INPUT_MEANS[3]) / INPUT_STDS[3],
        ((float)log(inputs[4] * 1.331 + 1.0) - INPUT_MEANS[4]) / INPUT_STDS[4],
    };

    // input layer
    alignas(64) float buffer1[LAYER_SIZES[0]] = {};
    alignas(64) float buffer2[LAYER_SIZES[1]] = {};
    mlpLayer(scaled_inputs, buffer1, W0, B0, N_INPUTS, LAYER_SIZES[0]);
    mlpLayer(buffer1, buffer2, W1, B1, LAYER_SIZES[0], LAYER_SIZES[1]);
    return outputLayer(buffer2, W2, B2[0], LAYER_SIZES[1]);
}
