#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include "cuda_kernels.cuh"
#include "labeling.h"
#include "window_2d.h"

void labelImage(Window2D *window, bool sufficientPrecision)
{
  computeJuliaCuda(window->labelSize, window->labelSize, window->c,
                   window->zoomLevel, window->xOffset, window->yOffset,
                   window->dLabelImage, window->streams[0]);
  CUDA_CHECK(cudaStreamSynchronize(window->streams[0]));
  CUDA_CHECK(cudaMemcpy(window->hLabelImage, window->dLabelImage,
                        window->labelSize * window->labelSize * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // save image
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  if (sufficientPrecision)
    ss << "../precision_labeling/sufficient2312/";
  else
    ss << "../precision_labeling/insufficient2312/";
  ss << now_c << ".pfm";
  std::string filename = ss.str();

  std::ofstream file(filename, std::ios::binary | std::ofstream::trunc);
  if (!file)
  {
    std::cerr << "Could not open file for writing." << std::endl;
    return;
  }

  file << "Pf\n";
  file << window->labelSize << " " << window->labelSize << "\n";
  file << "-1.0\n";
  file.write(reinterpret_cast<const char *>(window->hLabelImage),
             window->labelSize * window->labelSize * sizeof(float));
  file.close();

  // save params
  std::string csvName;
  if (sufficientPrecision)
    csvName = std::string("../precision_labeling/sufficient2312.csv");
  else
    csvName = std::string("../precision_labeling/insufficient2312.csv");

  std::ofstream csvFile(csvName, std::ios::app);
  if (!csvFile)
  {
    std::cerr << "Could not open csv file." << std::endl;
  }
  csvFile << filename << "," << window->c.real() << "," << window->c.imag()
          << "," << window->xOffset << "," << window->yOffset << ","
          << window->zoomLevel << "\n";
  csvFile.close();
}