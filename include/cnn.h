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

    CNNModel(std::string engineName, std::string onnxName)
    {
        if (!std::filesystem::exists(engineName))
            buildEngine(onnxName, engineName);
        loadEngine(engineName);

        CUDA_CHECK(cudaMallocHost((void **)&hOutBuf, sizeof(float)));

        CUDA_CHECK(cudaMalloc((void **)&dInBuf, imgSize * imgSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&dOutBuf, sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&stream));

        context->setTensorAddress("input", dInBuf);
        context->setTensorAddress("output", dOutBuf);

        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        context->enqueueV3(stream);
        CUDA_CHECK(cudaMemcpyAsync(hOutBuf, dOutBuf, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphInstance, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    ~CNNModel()
    {
        CUDA_CHECK(cudaFreeHost(hOutBuf));
        CUDA_CHECK(cudaFree(dInBuf));
        CUDA_CHECK(cudaFree(dOutBuf));

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaGraphExecDestroy(graphInstance));

        delete context;
        delete engine;
    }

    void enqueue(std::complex<double> c, double zoomLevel, double xOffset, double yOffset)
    {
        computeJuliaCuda(imgSize, imgSize, c, zoomLevel, xOffset, yOffset, dInBuf, stream);
        CUDA_CHECK(cudaGraphLaunch(graphInstance, stream));
    }

    float getPred()
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

    bool readFile(const std::string &fileName, std::vector<char> &buffer)
    {
        std::ifstream file(fileName, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << fileName << std::endl;
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        buffer.resize(size);
        if (!file.read(buffer.data(), size))
        {
            std::cerr << "Error reading file: " << fileName << std::endl;
            return false;
        }
        return true;
    }

    bool buildEngine(std::string &onnxName, std::string &engineName)
    {
        IBuilder *builder = createInferBuilder(logger);
        INetworkDefinition *network = builder->createNetworkV2(1U);

        IParser *parser = createParser(*network, logger);
        std::vector<char> modelData;
        if (!readFile(onnxName, modelData))
            return false;

        if (!parser->parse(modelData.data(), modelData.size()))
        {
            for (int i = 0; i < parser->getNbErrors(); ++i)
                std::cout << parser->getError(i)->desc() << std::endl;
            return false;
        }

        IBuilderConfig *config = builder->createBuilderConfig();
        IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);

        std::ofstream file(engineName, std::ios::binary);
        if (!file)
        {
            std::cerr << "Could not open engine file for writing." << std::endl;
            return false;
        }
        file.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());
        file.close();

        delete parser;
        delete network;
        delete config;
        delete builder;
        delete serializedModel;

        return true;
    }

    void loadEngine(std::string &engineName)
    {
        IRuntime *runtime = createInferRuntime(logger);
        std::vector<char> modelData;
        readFile(engineName, modelData);
        engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
        context = engine->createExecutionContext();

        delete runtime;
    }
};
