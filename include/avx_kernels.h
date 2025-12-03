#pragma once

#include <complex>

void computeJuliaAvx(int width, int height, std::complex<double> c,
                     double zoomLevel, double xOffset, double yOffset,
                     float* buffer);
