#pragma once

#include "argus_inference/cuda_kernels.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace argus_inference {

struct SegmentationInstance {
    int classId = -1;
    std::string className;
    float confidence = 0.0F;
    cv::Rect box;
    cv::Mat mask;
};

struct InferenceTiming {
    float gpuPreprocessMs = 0.0F;
    float tensorRtMs = 0.0F;
    float outputCopyMs = 0.0F;
    float candidateDecodeMs = 0.0F;
    float nmsMs = 0.0F;
    float maskDecodeMs = 0.0F;
    float totalMs = 0.0F;
};

class YoloV8Segmentation final {
public:
    YoloV8Segmentation();
    ~YoloV8Segmentation();

    YoloV8Segmentation(const YoloV8Segmentation&) = delete;
    YoloV8Segmentation& operator=(const YoloV8Segmentation&) = delete;

    bool initialize(const std::string& enginePath, int inputSize, bool requireFp16);
    bool infer(const void* rgbaDevice, size_t sourcePitch, int sourceWidth, int sourceHeight,
               int rgbaWidth, int rgbaHeight, float confidenceThreshold, float iouThreshold,
               std::vector<SegmentationInstance>* instances, InferenceTiming* timing);
    bool synchronize() const;

    int inputSize() const { return inputSize_; }

private:
    class Logger final : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override;
    };

    struct Tensor {
        std::string name;
        nvinfer1::Dims dims{};
        void* device = nullptr;
        size_t bytes = 0;
    };

    bool loadEngine(const std::string& enginePath, bool requireFp16);
    bool initializeRuntime(const void* serialized, size_t size, bool requireFp16);
    bool configureTensors();
    bool usesFp16Precision() const;
    void releaseBuffers();
    void releaseTimingEvents();

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t gpuStartEvent_ = nullptr;
    cudaEvent_t gpuPreprocessEvent_ = nullptr;
    cudaEvent_t gpuInferenceEvent_ = nullptr;
    cudaEvent_t gpuOutputCopyEvent_ = nullptr;
    Tensor input_;
    std::vector<Tensor> outputs_;
    std::vector<std::vector<float>> hostOutputs_;
    GpuMaskDecoder maskDecoder_;
    int inputSize_ = 640;
    bool initialized_ = false;
};

}  // namespace argus_inference
