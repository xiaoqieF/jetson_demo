#pragma once

#include <argus_interfaces/msg/argus_inference_result.hpp>
#include <argus_interfaces/msg/argus_qwen_description.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace qwen_description {

class QwenDescriptionNode final : public rclcpp::Node {
public:
    explicit QwenDescriptionNode(const rclcpp::NodeOptions& options);
    ~QwenDescriptionNode() override;

private:
    struct Detection {
        std_msgs::msg::Header header;
        uint64_t frameNumber = 0;
        std::vector<argus_interfaces::msg::ArgusInstanceSegmentation> instances;
    };

    struct Job {
        sensor_msgs::msg::CompressedImage::ConstSharedPtr image;
        Detection detection;
    };

    void onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr image);
    void onDetection(argus_interfaces::msg::ArgusInferenceResult::ConstSharedPtr result);
    void workerLoop();
    void process(Job job);
    trt_edgellm::rt::imageUtils::ImageData makeImage(
        const sensor_msgs::msg::CompressedImage& image);
    std::string buildPrompt(const Detection& detection) const;
    bool isTarget(const argus_interfaces::msg::ArgusInstanceSegmentation& instance) const;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr imageSubscription_;
    rclcpp::Subscription<argus_interfaces::msg::ArgusInferenceResult>::SharedPtr detectionSubscription_;
    rclcpp::Publisher<argus_interfaces::msg::ArgusQwenDescription>::SharedPtr publisher_;

    Detection latestDetection_;
    bool haveDetection_ = false;
    bool warnedTargetClassFallback_ = false;
    uint64_t imageCount_ = 0;
    std::mutex cacheMutex_;
    std::optional<Job> pendingJob_;
    std::mutex jobMutex_;
    std::condition_variable jobReady_;
    std::thread workerThread_;
    bool stopWorker_ = false;

    std::string inputTopic_;
    std::string detectionTopic_;
    std::string outputTopic_;
    std::string promptPrefix_;
    std::vector<std::string> targetClasses_;
    double minConfidence_ = 0.4;
    int maxGenerateLength_ = 128;
    double temperature_ = 0.0;
    uint64_t inferenceEveryNFrames_ = 100;
    std::string engineDir_;
    std::string multimodalEngineDir_;

    cudaStream_t stream_ = nullptr;
    decltype(trt_edgellm::loadEdgellmPluginLib()) plugin_;
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime_;
};

}  // namespace qwen_description
