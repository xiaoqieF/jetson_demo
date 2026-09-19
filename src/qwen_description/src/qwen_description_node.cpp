#include "qwen_description/qwen_description_node.hpp"

#include <NvBufSurface.h>
#include <nvbufsurface.h>

#include <algorithm>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace qwen_description {
namespace {

std::string trimRepetitiveTail(std::string text) {
    constexpr size_t kMinimumPatternBytes = 12;
    constexpr size_t kRepeatCount = 3;

    for (size_t start = 0; start + kMinimumPatternBytes * kRepeatCount <= text.size(); ++start) {
        const size_t maximumPatternBytes = (text.size() - start) / kRepeatCount;
        for (size_t patternBytes = kMinimumPatternBytes;
             patternBytes <= maximumPatternBytes; ++patternBytes) {
            const std::string_view pattern(text.data() + start, patternBytes);
            bool repeated = true;
            for (size_t repeat = 1; repeat < kRepeatCount; ++repeat) {
                if (std::string_view(text.data() + start + repeat * patternBytes, patternBytes) != pattern) {
                    repeated = false;
                    break;
                }
            }
            if (repeated) {
                text.erase(start + patternBytes);
                return text;
            }
        }
    }
    return text;
}

}  // namespace

QwenDescriptionNode::QwenDescriptionNode(const rclcpp::NodeOptions& options)
    : Node("qwen_description_node", options) {
    inputTopic_ = declare_parameter<std::string>("input_topic", "/camera/image/compressed");
    detectionTopic_ = declare_parameter<std::string>("detection_topic", "/camera/inference/result");
    outputTopic_ = declare_parameter<std::string>("output_topic", "/camera/inference/qwen_description");
    actionName_ = declare_parameter<std::string>("action_name", "/camera/inference/qwen");
    engineDir_ = declare_parameter<std::string>(
        "engine_dir", "/home/royfan/qwen3-vl-2b/engines/int4/llm");
    multimodalEngineDir_ = declare_parameter<std::string>(
        "multimodal_engine_dir", "/home/royfan/qwen3-vl-2b/engines/int4");
    targetClasses_ = declare_parameter<std::vector<std::string>>("target_classes", std::vector<std::string>{"person"});
    minConfidence_ = declare_parameter<double>("min_confidence", 0.4);
    maxGenerateLength_ = declare_parameter<int>("max_generate_length", 256);
    temperature_ = declare_parameter<double>("temperature", 0.0);
    if (engineDir_.empty() || multimodalEngineDir_.empty() || maxGenerateLength_ <= 0 ||
        actionName_.empty() || minConfidence_ < 0.0 || minConfidence_ > 1.0 || temperature_ < 0.0) {
        throw std::invalid_argument("Qwen description parameter is invalid");
    }

    plugin_ = trt_edgellm::loadEdgellmPluginLib();
    if (!plugin_) throw std::runtime_error("无法加载 TensorRT-LLM plugin library");
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
        throw std::runtime_error("无法创建 Qwen CUDA stream");
    }
    try {
        runtime_ = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
            engineDir_, multimodalEngineDir_,
            std::unordered_map<std::string, std::string>{}, stream_);
    } catch (...) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
        throw;
    }

    publisher_ = create_publisher<argus_interfaces::msg::ArgusQwenDescription>(
        outputTopic_, rclcpp::QoS(rclcpp::KeepLast(4)).best_effort());
    imageSubscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        inputTopic_, rclcpp::QoS(rclcpp::KeepLast(4)).best_effort(),
        [this](sensor_msgs::msg::CompressedImage::ConstSharedPtr image) { onImage(std::move(image)); });
    detectionSubscription_ = create_subscription<argus_interfaces::msg::ArgusInferenceResult>(
        detectionTopic_, rclcpp::QoS(rclcpp::KeepLast(4)).best_effort(),
        [this](argus_interfaces::msg::ArgusInferenceResult::ConstSharedPtr result) {
            onDetection(std::move(result));
        });
    actionServer_ = rclcpp_action::create_server<QwenInference>(
        this, actionName_,
        [this](const rclcpp_action::GoalUUID& uuid,
               std::shared_ptr<const QwenInference::Goal> goal) {
            return onGoal(uuid, std::move(goal));
        },
        [this](const std::shared_ptr<GoalHandleQwenInference> goalHandle) {
            return onCancel(goalHandle);
        },
        [this](const std::shared_ptr<GoalHandleQwenInference> goalHandle) {
            onAccepted(goalHandle);
        });
    workerThread_ = std::thread(&QwenDescriptionNode::workerLoop, this);
    RCLCPP_INFO(get_logger(), "Qwen3-VL runtime loaded from %s; action ready at %s",
                engineDir_.c_str(), actionName_.c_str());
}

QwenDescriptionNode::~QwenDescriptionNode() {
    actionServer_.reset();
    imageSubscription_.reset();
    detectionSubscription_.reset();
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        stopWorker_ = true;
    }
    jobReady_.notify_one();
    if (workerThread_.joinable()) workerThread_.join();
    runtime_.reset();
    if (stream_) cudaStreamDestroy(stream_);
}

void QwenDescriptionNode::onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr image) {
    if (!image || image->data.empty()) return;
    std::lock_guard<std::mutex> lock(cacheMutex_);
    latestImage_ = std::move(image);
}

void QwenDescriptionNode::onDetection(
    argus_interfaces::msg::ArgusInferenceResult::ConstSharedPtr result) {
    if (!result) return;
    Detection detection{result->header, result->frame_number, {}};
    for (const auto& instance : result->instances) {
        if (isTarget(instance)) detection.instances.push_back(instance);
    }
    if (detection.instances.empty() && !result->instances.empty()) {
        detection.instances = result->instances;
        if (!warnedTargetClassFallback_) {
            warnedTargetClassFallback_ = true;
            RCLCPP_WARN(get_logger(), "检测结果未匹配 target_classes，后续将回退使用全部实例");
        }
    }
    std::lock_guard<std::mutex> lock(cacheMutex_);
    latestDetection_ = std::move(detection);
    haveDetection_ = !latestDetection_.instances.empty();
}

bool QwenDescriptionNode::isTarget(
    const argus_interfaces::msg::ArgusInstanceSegmentation& instance) const {
    if (instance.confidence < minConfidence_) return false;
    return targetClasses_.empty() ||
        std::find(targetClasses_.begin(), targetClasses_.end(), instance.class_name) != targetClasses_.end();
}

std::string QwenDescriptionNode::buildPrompt(
    const std::string& requestPrompt, const Detection& detection) const {
    std::ostringstream prompt;
    prompt << requestPrompt << "\n检测候选目标：\n";
    for (size_t index = 0; index < detection.instances.size(); ++index) {
        const auto& item = detection.instances[index];
        prompt << index + 1 << ". " << item.class_name << "，置信度 " << item.confidence
               << "，边界框 [" << item.x_min << ", " << item.y_min << ", "
               << item.x_max << ", " << item.y_max << "]\n";
    }
    prompt << "请先自行核验，再给出最终描述。";
    return prompt.str();
}

trt_edgellm::rt::imageUtils::ImageData QwenDescriptionNode::makeImage(
    const sensor_msgs::msg::CompressedImage& image) {
    cv::Mat rgb = cv::imdecode(image.data, cv::IMREAD_COLOR);
    if (rgb.empty()) throw std::runtime_error("JPEG 解码失败");
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
    const auto width = static_cast<uint32_t>(rgb.cols);
    const auto height = static_cast<uint32_t>(rgb.rows);
    trt_edgellm::rt::Tensor tensor({1, height, width, 3},
        trt_edgellm::rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "qwen_full_frame");
    for (uint32_t row = 0; row < height; ++row)
        std::memcpy(static_cast<uint8_t*>(tensor.rawPointer()) + static_cast<size_t>(row) * width * 3,
                    rgb.ptr(static_cast<int>(row)), static_cast<size_t>(width) * 3);
    trt_edgellm::rt::imageUtils::ImageData result(std::move(tensor));
    result.doResize = true;
    return result;
}

rclcpp_action::GoalResponse QwenDescriptionNode::onGoal(
    const rclcpp_action::GoalUUID&,
    std::shared_ptr<const QwenInference::Goal> goal) {
    if (!goal || goal->prompt.empty()) {
        RCLCPP_WARN(get_logger(), "拒绝空 prompt 的 Qwen goal");
        return rclcpp_action::GoalResponse::REJECT;
    }
    bool expected = false;
    if (!goalActive_.compare_exchange_strong(expected, true)) {
        RCLCPP_WARN(get_logger(), "Qwen 正在推理，拒绝新的 goal");
        return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse QwenDescriptionNode::onCancel(
    const std::shared_ptr<GoalHandleQwenInference>) {
    return rclcpp_action::CancelResponse::ACCEPT;
}

void QwenDescriptionNode::onAccepted(
    const std::shared_ptr<GoalHandleQwenInference> goalHandle) {
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        pendingGoal_ = goalHandle;
    }
    jobReady_.notify_one();
}

void QwenDescriptionNode::workerLoop() {
    while (true) {
        std::shared_ptr<GoalHandleQwenInference> goalHandle;
        {
            std::unique_lock<std::mutex> lock(jobMutex_);
            jobReady_.wait(lock, [this] { return stopWorker_ || pendingGoal_.has_value(); });
            if (stopWorker_ && !pendingGoal_) return;
            goalHandle = std::move(*pendingGoal_);
            pendingGoal_.reset();
        }
        processGoal(goalHandle);
        goalActive_.store(false);
    }
}

void QwenDescriptionNode::processGoal(
    const std::shared_ptr<GoalHandleQwenInference>& goalHandle) {
    const auto goal = goalHandle->get_goal();
    auto result = std::make_shared<QwenInference::Result>();
    auto& output = result->result;
    auto publishStage = [&goalHandle](const std::string& stage) {
        auto feedback = std::make_shared<QwenInference::Feedback>();
        feedback->stage = stage;
        goalHandle->publish_feedback(feedback);
    };

    if (goalHandle->is_canceling()) {
        output.error_message = "推理已取消";
        goalHandle->canceled(result);
        return;
    }
    publishStage("preparing");

    sensor_msgs::msg::CompressedImage::ConstSharedPtr image;
    Detection detection;
    bool haveDetection = false;
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        image = latestImage_;
        detection = latestDetection_;
        haveDetection = haveDetection_;
    }
    if (!image) {
        output.error_message = "尚未收到可用于推理的图像";
        goalHandle->abort(result);
        return;
    }
    output.header = image->header;
    if (haveDetection) {
        output.frame_number = detection.frameNumber;
        for (const auto& instance : detection.instances) {
            output.candidate_classes.push_back(instance.class_name);
        }
    }

    const auto start = std::chrono::steady_clock::now();
    try {
        publishStage("inferencing");
        trt_edgellm::rt::Message message;
        message.role = "user";
        message.contents.push_back({"image", ""});
        std::string prompt = goal->include_detection_context && haveDetection
            ? buildPrompt(goal->prompt, detection)
            : goal->prompt;
        prompt += "\n请只输出最终描述，不要复述提示词或分析过程。使用简明、准确、完整的中文，"
                  "不超过 200 字，避免重复任何句子。";
        message.contents.push_back({"text", prompt});
        trt_edgellm::rt::LLMGenerationRequest generationRequest;
        generationRequest.requests.resize(1);
        generationRequest.requests[0].messages.push_back(std::move(message));
        generationRequest.requests[0].imageBuffers.push_back(makeImage(*image));
        generationRequest.temperature = static_cast<float>(temperature_);
        generationRequest.topP = 1.0F;
        generationRequest.topK = 1;
        generationRequest.maxGenerateLength = maxGenerateLength_;
        trt_edgellm::rt::LLMGenerationResponse generationResponse;
        if (!runtime_->handleRequest(generationRequest, generationResponse, stream_) ||
            generationResponse.outputTexts.empty()) {
            throw std::runtime_error("Qwen runtime handleRequest failed");
        }
        output.description = trimRepetitiveTail(generationResponse.outputTexts.front());
        output.success = true;
    } catch (const std::exception& error) {
        output.success = false;
        output.error_message = error.what();
        RCLCPP_ERROR(get_logger(), "Qwen full-frame inference failed: %s", error.what());
    }
    output.inference_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
    if (output.success) {
        RCLCPP_INFO(get_logger(), "Qwen 推理完成：frame=%lu，候选目标=%zu，耗时=%.2f ms",
                    static_cast<unsigned long>(output.frame_number),
                    output.candidate_classes.size(), output.inference_ms);
    } else {
        RCLCPP_WARN(get_logger(), "Qwen 推理失败：frame=%lu，耗时=%.2f ms，错误：%s",
                    static_cast<unsigned long>(output.frame_number), output.inference_ms,
                    output.error_message.c_str());
    }
    if (goalHandle->is_canceling()) {
        output.success = false;
        output.error_message = "推理已取消";
        publisher_->publish(output);
        goalHandle->canceled(result);
    } else if (output.success) {
        publisher_->publish(output);
        goalHandle->succeed(result);
    } else {
        publisher_->publish(output);
        goalHandle->abort(result);
    }
}

}  // namespace qwen_description

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(qwen_description::QwenDescriptionNode)
