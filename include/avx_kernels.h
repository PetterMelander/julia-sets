#pragma once

void compute_julia_avx(int width, int height, double c_re, double c_im,
                       double zoomLevel, double x_offset, double y_offset,
                       float *buffer);
// void julia(float *intensities, double range, double x_offset, double
// y_offset,
//            double c_re, double c_im, int width, int height);