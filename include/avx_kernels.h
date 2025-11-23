#pragma once

#include "gl_utils.h"

void compute_julia_avx(ProgramState state, float *buffer);
void julia(float *intensities, double range, double x_offset, double y_offset,
           double c_re, double c_im, int width, int height);