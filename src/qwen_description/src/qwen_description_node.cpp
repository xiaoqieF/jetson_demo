#include "qwen_description/qwen_description_node.hpp"

#include <NvBufSurface.h>
#include <nvbufsurface.h>

#include <algorithm>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qwen_description {

QwenDescriptionNode::QwenDescriptionNode(const rclcpp::NodeOptions& options)
    : Node("qwen_description_node", options) {
    inputTopic_ = declare_parameter<std::string>("input_topic", "/camera/image/compressed");
    detectionTopic_ = declare_parameter<std::string>("detection_topic", "/camera/inference/result");
    outputTopic_ = declare_parameter<std::string>("output_topic", "/camera/inference/qwen_description");
    engineDir_ = declare_parameter<std::string>(
        "engine_dir", "/home/royfan/qwen3-vl-2b/engines/int4/llm");
    multimodalEngineDir_ = declare_parameter<std::string>(
        "multimodal_engine_dir", "/home/royfan/qwen3-vl-2b/engines/int4");
    promptPrefix_ = declare_parameter<std::string>(
        "prompt", "请查看整张图片，核验检测候选目标是否真实存在，并用中文简洁描述确认存在的目标及其周围环境。不要盲目相信检测结果，不要推测无法从图像确认的事实。");
    targetClasses_ = declare_parameter<std::vector<std::string>>("target_classes", std::vector<std::string>{"person"});
    minConfidence_ = declare_parameter<double>("min_confidence", 0.4);
    maxGenerateLength_ = declare_parameter<int>("max_generate_length", 128);
    temperature_ = declare_parameter<double>("temperature", 0.0);
    inferenceEveryNFrames_ = static_cast<uint64_t>(declare_parameter<int>("inference_every_n_frames", 100));
    if (engineDir_.empty() || multimodalEngineDir_.empty() || maxGenerateLength_ <= 0 ||
        minConfidence_ < 0.0 || minConfidence_ > 1.0 || temperature_ < 0.0 || inferenceEveryNFrames_ == 0) {
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
    workerThread_ = std::thread(&QwenDescriptionNode::workerLoop, this);
    RCLCPP_INFO(get_logger(), "Qwen3-VL runtime loaded from %s", engineDir_.c_str());
}

QwenDescriptionNode::~QwenDescriptionNode() {
    imageSubscription_.reset();
    detectionSubscription_.reset();
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        stopWorker_ = true;
        pendingJob_.reset();
    }
    jobReady_.notify_one();
    if (workerThread_.joinable()) workerThread_.join();
    runtime_.reset();
    if (stream_) cudaStreamDestroy(stream_);
}

void QwenDescriptionNode::onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr image) {
    if (!image || image->data.empty()) return;
    std::lock_guard<std::mutex> lock(cacheMutex_);
    ++imageCount_;
    if (imageCount_ % inferenceEveryNFrames_ != 0) return;
    RCLCPP_INFO(get_logger(), "Qwen JPEG 采样：image_count=%lu，已有检测=%s",
                static_cast<unsigned long>(imageCount_), haveDetection_ ? "是" : "否");
    if (!haveDetection_) return;
    std::lock_guard<std::mutex> jobLock(jobMutex_);
    if (stopWorker_) return;
    pendingJob_ = Job{std::move(image), latestDetection_};
    jobReady_.notify_one();
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
    if (detection.instances.empty()) return;
    std::lock_guard<std::mutex> lock(cacheMutex_);
    latestDetection_ = std::move(detection);
    haveDetection_ = true;
}

bool QwenDescriptionNode::isTarget(
    const argus_interfaces::msg::ArgusInstanceSegmentation& instance) const {
    if (instance.confidence < minConfidence_) return false;
    return targetClasses_.empty() ||
        std::find(targetClasses_.begin(), targetClasses_.end(), instance.class_name) != targetClasses_.end();
}

std::string QwenDescriptionNode::buildPrompt(const Detection& detection) const {
    std::ostringstream prompt;
    prompt << promptPrefix_ << "\n检测候选目标：\n";
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

void QwenDescriptionNode::workerLoop() {
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(jobMutex_);
            jobReady_.wait(lock, [this] { return stopWorker_ || pendingJob_.has_value(); });
            if (stopWorker_) return;
            job = std::move(*pendingJob_);
            pendingJob_.reset();
        }
        process(std::move(job));
    }
}

void QwenDescriptionNode::process(Job job) {
    argus_interfaces::msg::ArgusQwenDescription output;
    output.header = job.detection.header;
    output.frame_number = job.detection.frameNumber;
    for (const auto& instance : job.detection.instances) output.candidate_classes.push_back(instance.class_name);
    const auto start = std::chrono::steady_clock::now();
    try {
        trt_edgellm::rt::LLMGenerationRequest request;
        trt_edgellm::rt::Message message;
        message.role = "user";
        message.contents.push_back({"image", ""});
        message.contents.push_back({"text", buildPrompt(job.detection)});
        request.requests.resize(1);
        request.requests[0].messages.push_back(std::move(message));
        request.requests[0].imageBuffers.push_back(makeImage(*job.image));
        request.temperature = static_cast<float>(temperature_);
        request.topP = 1.0F;
        request.topK = 1;
        request.maxGenerateLength = maxGenerateLength_;
        trt_edgellm::rt::LLMGenerationResponse response;
        if (!runtime_->handleRequest(request, response, stream_) || response.outputTexts.empty()) {
            throw std::runtime_error("Qwen runtime handleRequest failed");
        }
        output.description = response.outputTexts.front();
        output.success = true;
        RCLCPP_INFO(get_logger(), "Qwen 推理完成：frame=%lu，候选目标=%zu，耗时=%.2f ms，描述：%s",
                    static_cast<unsigned long>(output.frame_number),
                    output.candidate_classes.size(), output.inference_ms,
                    output.description.c_str());
    } catch (const std::exception& error) {
        output.success = false;
        output.error_message = error.what();
        RCLCPP_ERROR(get_logger(), "Qwen full-frame inference failed: %s", error.what());
    }
    output.inference_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
    if (!output.success) {
        RCLCPP_WARN(get_logger(), "Qwen 推理失败：frame=%lu，耗时=%.2f ms，错误：%s",
                    static_cast<unsigned long>(output.frame_number), output.inference_ms,
                    output.error_message.c_str());
    }
    publisher_->publish(std::move(output));
}

}  // namespace qwen_description

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(qwen_description::QwenDescriptionNode)
