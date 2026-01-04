#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "cuda_kernels.cuh"

using namespace nvonnxparser;
using namespace nvinfer1;

class CNNModel
{
public:
    static constexpr int imgSize = 224;

    CNNModel(std::string engineName, std::string onnxName);

    ~CNNModel();

    inline void enqueue(std::complex<double> c, double zoomLevel, double xOffset, double yOffset)
    {
        computeJuliaCuda(imgSize, imgSize, c, zoomLevel, xOffset, yOffset, dInBuf, stream);
        CUDA_CHECK(cudaGraphLaunch(graphInstance, stream));
    }

    inline float getPred()
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return *hOutBuf;
    }

private:
    ICudaEngine *engine;
    IExecutionContext *context;

    cudaStream_t stream;
    cudaGraphExec_t graphInstance;

    float *hOutBuf;
    float *dInBuf;
    float *dOutBuf;

    class Logger : public ILogger
    {
        void log(Severity severity, const char *msg) noexcept override
        {
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    };
    inline static Logger logger;

    bool readFile(const std::string &fileName, std::vector<char> &buffer);

    bool buildEngine(std::string &onnxName, std::string &engineName);

    void loadEngine(std::string &engineName);
};
