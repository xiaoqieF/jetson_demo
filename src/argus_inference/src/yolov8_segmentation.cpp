#include "argus_inference/yolov8_segmentation.hpp"

#include "argus_inference/cuda_preprocess.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace argus_inference {
namespace {

constexpr int kMaskThreshold = 128;

const std::vector<std::string> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

size_t volume(const nvinfer1::Dims& dims) {
    size_t result = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) return 0;
        result *= static_cast<size_t>(dims.d[index]);
    }
    return result;
}

bool hasDynamicDimension(const nvinfer1::Dims& dims) {
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) return true;
    }
    return false;
}

std::string className(int classId) {
    if (classId >= 0 && classId < static_cast<int>(kCocoClasses.size())) {
        return kCocoClasses[classId];
    }
    return "class_" + std::to_string(classId);
}

struct LetterboxInfo {
    float scaleX = 1.0F;
    float scaleY = 1.0F;
    int paddingX = 0;
    int paddingY = 0;
};

float outputAt(const std::vector<float>& output, int predictions, int prediction, int attribute) {
    return output[static_cast<size_t>(attribute) * predictions + prediction];
}

}  // namespace

YoloV8Segmentation::YoloV8Segmentation() {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        stream_ = nullptr;
        return;
    }
    if (cudaEventCreate(&gpuStartEvent_) != cudaSuccess ||
        cudaEventCreate(&gpuPreprocessEvent_) != cudaSuccess ||
        cudaEventCreate(&gpuInferenceEvent_) != cudaSuccess ||
        cudaEventCreate(&gpuOutputCopyEvent_) != cudaSuccess) {
        releaseTimingEvents();
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

YoloV8Segmentation::~YoloV8Segmentation() {
    releaseBuffers();
    releaseTimingEvents();
    if (stream_) cudaStreamDestroy(stream_);
}

void YoloV8Segmentation::Logger::log(Severity severity, const char* message) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << message << '\n';
    }
}

bool YoloV8Segmentation::initialize(const std::string& enginePath, int inputSize, bool requireFp16) {
    if (inputSize <= 0 || !stream_ || !gpuStartEvent_ || !gpuPreprocessEvent_ ||
        !gpuInferenceEvent_ || !gpuOutputCopyEvent_) return false;
    inputSize_ = inputSize;
    return loadEngine(enginePath, requireFp16);
}

bool YoloV8Segmentation::loadEngine(const std::string& enginePath, bool requireFp16) {
    std::ifstream input(enginePath, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const auto size = input.tellg();
    if (size <= 0) return false;
    std::vector<char> serialized(static_cast<size_t>(size));
    input.seekg(0);
    input.read(serialized.data(), size);
    return input.good() && initializeRuntime(serialized.data(), serialized.size(), requireFp16);
}

bool YoloV8Segmentation::initializeRuntime(const void* serialized, size_t size, bool requireFp16) {
    releaseBuffers();
    context_.reset();
    engine_.reset();
    runtime_.reset();
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(serialized, size));
    if (!engine_) return false;
    const bool hasVerifiedFp16Precision = usesFp16Precision();
    if (requireFp16 && !hasVerifiedFp16Precision) {
        std::cerr << "TensorRT engine 未包含可验证的 FP16 计算层；请使用 --fp16 --profilingVerbosity=detailed 重新生成 engine\n";
        return false;
    }
    if (hasVerifiedFp16Precision) {
        std::cerr << "TensorRT engine 已验证包含 FP16 计算层\n";
    } else {
        std::cerr << "TensorRT engine 精度无法从 plan 元数据验证，将继续加载；"
                     "设置 require_fp16_engine:=true 可强制拒绝未验证 engine\n";
    }
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    return context_ && configureTensors();
}

bool YoloV8Segmentation::usesFp16Precision() const {
    if (!engine_ || engine_->getProfilingVerbosity() != nvinfer1::ProfilingVerbosity::kDETAILED) {
        std::cerr << "无法检查 TensorRT engine 精度：engine 未使用 detailed profiling verbosity 构建\n";
        return false;
    }
    const auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(engine_->createEngineInspector());
    if (!inspector) return false;
    const char* information = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    if (!information) return false;
    const std::string details(information);
    return details.find("\"Precision\": \"FP16\"") != std::string::npos ||
           details.find("\"Precision\":\"FP16\"") != std::string::npos ||
           details.find("Precision: FP16") != std::string::npos ||
           details.find("\"Format/Datatype\": \"Half\"") != std::string::npos ||
           details.find("\"Type\": \"Half\"") != std::string::npos;
}

bool YoloV8Segmentation::configureTensors() {
    input_ = Tensor{};
    outputs_.clear();
    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
        const char* name = engine_->getIOTensorName(index);
        if (!name || engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            std::cerr << "仅支持 float32 输入/输出 TensorRT 张量\n";
            return false;
        }
        Tensor tensor{name, engine_->getTensorShape(name)};
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_ = std::move(tensor);
        } else {
            outputs_.push_back(std::move(tensor));
        }
    }
    if (input_.name.empty() || outputs_.size() < 2) return false;
    if (hasDynamicDimension(input_.dims) &&
        !context_->setInputShape(input_.name.c_str(), nvinfer1::Dims4{1, 3, inputSize_, inputSize_})) {
        return false;
    }
    input_.dims = context_->getTensorShape(input_.name.c_str());
    input_.bytes = volume(input_.dims) * sizeof(float);
    if (input_.bytes == 0 || cudaMalloc(&input_.device, input_.bytes) != cudaSuccess) return false;
    for (auto& output : outputs_) {
        output.dims = context_->getTensorShape(output.name.c_str());
        output.bytes = volume(output.dims) * sizeof(float);
        if (output.bytes == 0 || cudaMalloc(&output.device, output.bytes) != cudaSuccess) return false;
    }
    hostOutputs_.resize(outputs_.size());
    for (size_t index = 0; index < outputs_.size(); ++index) {
        hostOutputs_[index].resize(outputs_[index].bytes / sizeof(float));
    }
    initialized_ = true;
    return true;
}

void YoloV8Segmentation::releaseBuffers() {
    if (input_.device) cudaFree(input_.device);
    input_ = Tensor{};
    for (auto& output : outputs_) {
        if (output.device) cudaFree(output.device);
    }
    outputs_.clear();
    hostOutputs_.clear();
    initialized_ = false;
}

void YoloV8Segmentation::releaseTimingEvents() {
    if (gpuOutputCopyEvent_) cudaEventDestroy(gpuOutputCopyEvent_);
    gpuOutputCopyEvent_ = nullptr;
    if (gpuInferenceEvent_) cudaEventDestroy(gpuInferenceEvent_);
    gpuInferenceEvent_ = nullptr;
    if (gpuPreprocessEvent_) cudaEventDestroy(gpuPreprocessEvent_);
    gpuPreprocessEvent_ = nullptr;
    if (gpuStartEvent_) cudaEventDestroy(gpuStartEvent_);
    gpuStartEvent_ = nullptr;
}

bool YoloV8Segmentation::infer(const void* rgbaDevice, size_t sourcePitch, int sourceWidth,
                                int sourceHeight, int rgbaWidth, int rgbaHeight,
                                float confidenceThreshold, float iouThreshold,
                                std::vector<SegmentationInstance>* instances, InferenceTiming* timing) {
    if (!instances || !initialized_ || !rgbaDevice || sourceWidth <= 0 || sourceHeight <= 0 ||
        rgbaWidth <= 0 || rgbaHeight <= 0 || confidenceThreshold < 0.0F || iouThreshold < 0.0F) return false;
    if (timing) *timing = {};
    const auto totalStart = std::chrono::steady_clock::now();
    instances->clear();
    const float scale = std::min(static_cast<float>(inputSize_) / sourceWidth,
                                 static_cast<float>(inputSize_) / sourceHeight);
    const LetterboxInfo letterbox{scale, scale,
                                  (inputSize_ - static_cast<int>(std::round(sourceWidth * scale))) / 2,
                                  (inputSize_ - static_cast<int>(std::round(sourceHeight * scale))) / 2};
    if (cudaEventRecord(gpuStartEvent_, stream_) != cudaSuccess) return false;
    if (!preprocessRgbaOnGpu(rgbaDevice, sourcePitch, rgbaWidth, rgbaHeight,
                             static_cast<float*>(input_.device), inputSize_, stream_) ||
        cudaEventRecord(gpuPreprocessEvent_, stream_) != cudaSuccess ||
        !context_->setInputTensorAddress(input_.name.c_str(), input_.device)) return false;
    for (const auto& output : outputs_) {
        if (!context_->setOutputTensorAddress(output.name.c_str(), output.device)) return false;
    }
    if (!context_->enqueueV3(stream_)) return false;
    if (cudaEventRecord(gpuInferenceEvent_, stream_) != cudaSuccess) return false;
    for (size_t index = 0; index < outputs_.size(); ++index) {
        if (cudaMemcpyAsync(hostOutputs_[index].data(), outputs_[index].device, outputs_[index].bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess) return false;
    }
    if (cudaEventRecord(gpuOutputCopyEvent_, stream_) != cudaSuccess ||
        cudaEventSynchronize(gpuOutputCopyEvent_) != cudaSuccess) return false;
    float gpuPreprocessMs = 0.0F;
    float tensorRtMs = 0.0F;
    float outputCopyMs = 0.0F;
    if (cudaEventElapsedTime(&gpuPreprocessMs, gpuStartEvent_, gpuPreprocessEvent_) != cudaSuccess ||
        cudaEventElapsedTime(&tensorRtMs, gpuPreprocessEvent_, gpuInferenceEvent_) != cudaSuccess ||
        cudaEventElapsedTime(&outputCopyMs, gpuInferenceEvent_, gpuOutputCopyEvent_) != cudaSuccess) return false;
    if (timing) {
        timing->gpuPreprocessMs = gpuPreprocessMs;
        timing->tensorRtMs = tensorRtMs;
        timing->outputCopyMs = outputCopyMs;
    }

    int headIndex = -1;
    int protoIndex = -1;
    for (size_t index = 0; index < outputs_.size(); ++index) {
        if (outputs_[index].dims.nbDims == 3) headIndex = static_cast<int>(index);
        if (outputs_[index].dims.nbDims == 4) protoIndex = static_cast<int>(index);
    }
    if (headIndex < 0 || protoIndex < 0) return false;
    const auto& headDims = outputs_[headIndex].dims;
    const auto& protoDims = outputs_[protoIndex].dims;
    if (headDims.d[0] != 1 || headDims.d[1] <= 4 || headDims.d[2] <= 0 ||
        protoDims.d[0] != 1 || protoDims.d[1] <= 0 || protoDims.d[2] <= 0 || protoDims.d[3] <= 0) return false;
    const int attributes = headDims.d[1];
    const int predictions = headDims.d[2];
    const int maskDimensions = protoDims.d[1];
    const int classes = attributes - 4 - maskDimensions;
    if (classes <= 0) return false;

    struct Candidate { SegmentationInstance instance; std::vector<float> coefficients; };
    std::vector<Candidate> candidates;
    const auto candidateDecodeStart = std::chrono::steady_clock::now();
    const auto& head = hostOutputs_[headIndex];
    for (int prediction = 0; prediction < predictions; ++prediction) {
        int classId = 0;
        float confidence = outputAt(head, predictions, prediction, 4);
        for (int classIndex = 1; classIndex < classes; ++classIndex) {
            const float score = outputAt(head, predictions, prediction, 4 + classIndex);
            if (score > confidence) { confidence = score; classId = classIndex; }
        }
        if (confidence < confidenceThreshold) continue;
        const float centerX = outputAt(head, predictions, prediction, 0);
        const float centerY = outputAt(head, predictions, prediction, 1);
        const float width = outputAt(head, predictions, prediction, 2);
        const float height = outputAt(head, predictions, prediction, 3);
        const int left = std::clamp(static_cast<int>(std::floor((centerX - width * 0.5F - letterbox.paddingX) / letterbox.scaleX)), 0, sourceWidth);
        const int top = std::clamp(static_cast<int>(std::floor((centerY - height * 0.5F - letterbox.paddingY) / letterbox.scaleY)), 0, sourceHeight);
        const int right = std::clamp(static_cast<int>(std::ceil((centerX + width * 0.5F - letterbox.paddingX) / letterbox.scaleX)), 0, sourceWidth);
        const int bottom = std::clamp(static_cast<int>(std::ceil((centerY + height * 0.5F - letterbox.paddingY) / letterbox.scaleY)), 0, sourceHeight);
        if (right <= left || bottom <= top) continue;
        Candidate candidate;
        candidate.instance = SegmentationInstance{classId, className(classId), confidence,
            cv::Rect(left, top, right - left, bottom - top), {}};
        candidate.coefficients.resize(maskDimensions);
        for (int maskIndex = 0; maskIndex < maskDimensions; ++maskIndex) {
            candidate.coefficients[maskIndex] = outputAt(head, predictions, prediction, 4 + classes + maskIndex);
        }
        candidates.push_back(std::move(candidate));
    }
    if (timing) {
        timing->candidateDecodeMs = static_cast<float>(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - candidateDecodeStart).count());
    }
    if (candidates.empty()) {
        if (timing) {
            timing->totalMs = static_cast<float>(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - totalStart).count());
        }
        return true;
    }

    const auto nmsStart = std::chrono::steady_clock::now();
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    boxes.reserve(candidates.size());
    confidences.reserve(candidates.size());
    classIds.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        boxes.push_back(candidate.instance.box);
        confidences.push_back(candidate.instance.confidence);
        classIds.push_back(candidate.instance.classId);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxesBatched(boxes, confidences, classIds, confidenceThreshold, iouThreshold, keep);
    if (timing) {
        timing->nmsMs = static_cast<float>(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - nmsStart).count());
    }

    const auto maskDecodeStart = std::chrono::steady_clock::now();
    const auto& proto = hostOutputs_[protoIndex];
    cv::Mat protoMatrix(maskDimensions, protoDims.d[2] * protoDims.d[3], CV_32F,
                        const_cast<float*>(proto.data()));
    for (const int index : keep) {
        auto& candidate = candidates[index];
        cv::Mat coefficients(1, maskDimensions, CV_32F, candidate.coefficients.data());
        cv::Mat mask = coefficients * protoMatrix;
        mask = mask.reshape(1, protoDims.d[2]);
        cv::exp(-mask, mask);
        mask = 1.0F / (1.0F + mask);
        cv::resize(mask, mask, cv::Size(inputSize_, inputSize_), 0.0, 0.0, cv::INTER_LINEAR);
        const cv::Rect valid(letterbox.paddingX, letterbox.paddingY,
                             inputSize_ - 2 * letterbox.paddingX, inputSize_ - 2 * letterbox.paddingY);
        cv::resize(mask(valid), mask, cv::Size(sourceWidth, sourceHeight), 0.0, 0.0, cv::INTER_LINEAR);
        cv::threshold(mask(candidate.instance.box), candidate.instance.mask, 0.5, 255.0, cv::THRESH_BINARY);
        candidate.instance.mask.convertTo(candidate.instance.mask, CV_8U);
        instances->push_back(std::move(candidate.instance));
    }
    std::sort(instances->begin(), instances->end(), [](const auto& left, const auto& right) {
        return left.confidence > right.confidence;
    });
    if (timing) {
        timing->maskDecodeMs = static_cast<float>(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - maskDecodeStart).count());
        timing->totalMs = static_cast<float>(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - totalStart).count());
    }
    return true;
}

bool YoloV8Segmentation::synchronize() const {
    return !stream_ || cudaStreamSynchronize(stream_) == cudaSuccess;
}

}  // namespace argus_inference
