#pragma once

#include <argus_interfaces/action/qwen_inference.hpp>
#include <argus_interfaces/msg/argus_inference_result.hpp>
#include <argus_interfaces/msg/argus_qwen_description.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <condition_variable>
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
    using QwenInference = argus_interfaces::action::QwenInference;
    using GoalHandleQwenInference = rclcpp_action::ServerGoalHandle<QwenInference>;

    struct Detection {
        std_msgs::msg::Header header;
        uint64_t frameNumber = 0;
        std::vector<argus_interfaces::msg::ArgusInstanceSegmentation> instances;
    };

    void onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr image);
    void onDetection(argus_interfaces::msg::ArgusInferenceResult::ConstSharedPtr result);
    rclcpp_action::GoalResponse onGoal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const QwenInference::Goal> goal);
    rclcpp_action::CancelResponse onCancel(
        const std::shared_ptr<GoalHandleQwenInference> goalHandle);
    void onAccepted(const std::shared_ptr<GoalHandleQwenInference> goalHandle);
    void workerLoop();
    void processGoal(const std::shared_ptr<GoalHandleQwenInference>& goalHandle);
    trt_edgellm::rt::imageUtils::ImageData makeImage(
        const sensor_msgs::msg::CompressedImage& image);
    std::string buildPrompt(const std::string& prompt, const Detection& detection) const;
    bool isTarget(const argus_interfaces::msg::ArgusInstanceSegmentation& instance) const;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr imageSubscription_;
    rclcpp::Subscription<argus_interfaces::msg::ArgusInferenceResult>::SharedPtr detectionSubscription_;
    rclcpp::Publisher<argus_interfaces::msg::ArgusQwenDescription>::SharedPtr publisher_;
    rclcpp_action::Server<QwenInference>::SharedPtr actionServer_;

    sensor_msgs::msg::CompressedImage::ConstSharedPtr latestImage_;
    Detection latestDetection_;
    bool haveDetection_ = false;
    bool warnedTargetClassFallback_ = false;
    std::mutex cacheMutex_;
    std::optional<std::shared_ptr<GoalHandleQwenInference>> pendingGoal_;
    std::mutex jobMutex_;
    std::condition_variable jobReady_;
    std::thread workerThread_;
    std::atomic_bool goalActive_{false};
    bool stopWorker_ = false;

    std::string inputTopic_;
    std::string detectionTopic_;
    std::string outputTopic_;
    std::string actionName_;
    std::vector<std::string> targetClasses_;
    double minConfidence_ = 0.4;
    int maxGenerateLength_ = 256;
    double temperature_ = 0.0;
    std::string engineDir_;
    std::string multimodalEngineDir_;

    cudaStream_t stream_ = nullptr;
    decltype(trt_edgellm::loadEdgellmPluginLib()) plugin_;
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime_;
};

}  // namespace qwen_description
