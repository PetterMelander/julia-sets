#pragma once

#include <complex>

inline int mandelbrotConvergence(const std::complex<double> c)
{
    constexpr int MAX_ITER = 10000000;
    constexpr double ESCAPE_THRESHOLD = 2.0;
    
    const double cRe = c.real();
    const double cIm = c.imag();

    double zRe = 0.0;
    double zIm = 0.0;

    double zRe2 = 0.0;
    double zIm2 = 0.0;

    for (int i = 0; i < MAX_ITER; ++i)
    {
        if (zRe2 + zIm2 > ESCAPE_THRESHOLD * ESCAPE_THRESHOLD)
        {
            return i;
        }
        
        double tmp = 2.0 * zRe * zIm;
        zRe = zRe * zRe - zIm * zIm + cRe;
        zIm = tmp + cIm;

        zRe2 = zRe * zRe;
        zIm2 = zIm * zIm;
    }
    return MAX_ITER;
}

inline bool insideSet(const std::complex<double> c)
{
    double x = c.real();
    double y = c.imag();
    double y2 = y * y;

    bool insideBulb = (x + 1.0) * (x + 1.0) + y2 <= 1.0 / 16.0;
    double q = (x - 1.0 / 4.0) * (x - 1.0 / 4.0) + y2;
    bool insideCardiod = q * (q + (x - 1.0 / 4.0)) <= y2 / 4.0;
    return insideBulb || insideCardiod;
}
