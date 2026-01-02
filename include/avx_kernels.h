#pragma once

#include <complex>

constexpr int MAX_ITERS = 2500;

void computeJuliaAvx(int width, int height, std::complex<double> c, double zoomLevel,
                     double xOffset, double yOffset, float *buffer);
