#pragma once

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

class YoloV8Segmentation final {
public:
    YoloV8Segmentation();
    ~YoloV8Segmentation();

    YoloV8Segmentation(const YoloV8Segmentation&) = delete;
    YoloV8Segmentation& operator=(const YoloV8Segmentation&) = delete;

    bool initialize(const std::string& enginePath, int inputSize, bool requireFp16);
    bool infer(const void* rgbaDevice, size_t sourcePitch, int sourceWidth, int sourceHeight,
               int rgbaWidth, int rgbaHeight, float confidenceThreshold, float iouThreshold,
               std::vector<SegmentationInstance>* instances);

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

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    Tensor input_;
    std::vector<Tensor> outputs_;
    int inputSize_ = 640;
    bool initialized_ = false;
};

}  // namespace argus_inference
