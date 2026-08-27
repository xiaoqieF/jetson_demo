#include "argus_inference/inference_node.hpp"

#include "argus_inference/yolov8_segmentation.hpp"

#include <NvBufSurface.h>
#include <nvbufsurface.h>

#include <EGL/egl.h>
#include <cudaEGL.h>

#include <algorithm>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace argus_inference {

InferenceNode::InferenceNode(const rclcpp::NodeOptions& options)
    : Node("yuv_inference_node", options) {
    inputTopic_ = declare_parameter<std::string>("input_topic", "/camera/image/yuv");
    const auto outputTopic = declare_parameter<std::string>(
        "output_topic", "/camera/inference/overlay/compressed");
    const auto enginePath = declare_parameter<std::string>("engine_path", "/home/royfan/yolov8_trt/yolov8s-seg-640.engine");
    inputSize_ = declare_parameter<int>("input_size", 640);
    const auto requireFp16Engine = declare_parameter<bool>("require_fp16_engine", true);
    const auto timingLogEveryNFrames = declare_parameter<int>("timing_log_every_n_frames", 30);
    confidenceThreshold_ = static_cast<float>(declare_parameter<double>("confidence_threshold", 0.25));
    iouThreshold_ = static_cast<float>(declare_parameter<double>("iou_threshold", 0.45));
    overlayQuality_ = declare_parameter<int>("overlay_quality", 90);
    enableOverlay_ = declare_parameter<bool>("enable_overlay", false);
    if (inputSize_ <= 0 || timingLogEveryNFrames <= 0 ||
        confidenceThreshold_ < 0.0F || confidenceThreshold_ > 1.0F ||
        iouThreshold_ < 0.0F || iouThreshold_ > 1.0F || overlayQuality_ < 1 || overlayQuality_ > 100) {
        throw std::invalid_argument("TensorRT inference parameter is invalid");
    }
    timingLogEveryNFrames_ = static_cast<uint64_t>(timingLogEveryNFrames);

    model_ = std::make_unique<YoloV8Segmentation>();
    if (!model_->initialize(enginePath, inputSize_, requireFp16Engine)) {
        throw std::runtime_error("无法加载 YOLOv8 TensorRT engine: " + enginePath);
    }
    if (!initializeSlotSurface(&inferenceSlot_)) {
        releaseSlot(&inferenceSlot_);
        throw std::runtime_error("创建 RGBA 推理 dmabuf 失败");
    }
    if (enableOverlay_) {
        publisher_ = create_publisher<sensor_msgs::msg::CompressedImage>(
            outputTopic, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort());
    }
    inferenceThread_ = std::thread(&InferenceNode::inferenceLoop, this);
    subscription_ = create_subscription<argus_transport::ArgusFramePacket>(
        inputTopic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
        [this](argus_transport::ArgusFramePacket::ConstSharedPtr message) { stageFrame(std::move(message)); });
}

InferenceNode::~InferenceNode() {
    subscription_.reset();
    std::optional<PendingFrame> pendingFrame;
    {
        std::lock_guard<std::mutex> lock(jobsMutex_);
        stopInference_ = true;
        pendingFrame = std::move(pendingFrame_);
        pendingFrame_.reset();
    }
    pendingFrame.reset();
    jobsReady_.notify_one();
    if (inferenceThread_.joinable()) inferenceThread_.join();
    releaseSlot(&inferenceSlot_);
}

bool InferenceNode::initializeSlotSurface(StagingSlot* slot) {
    if (!slot) return false;
    NvBufSurf::NvCommonAllocateParams params{};
    params.width = static_cast<uint32_t>(inputSize_);
    params.height = static_cast<uint32_t>(inputSize_);
    params.memType = NVBUF_MEM_SURFACE_ARRAY;
    params.memtag = NvBufSurfaceTag_NONE;
    params.layout = NVBUF_LAYOUT_PITCH;
    params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    if (NvBufSurf::NvAllocate(&params, 1, &slot->rgbaDmabuf) < 0) {
        RCLCPP_ERROR(get_logger(), "创建 RGBA 推理 dmabuf 失败");
        slot->rgbaDmabuf = -1;
        return false;
    }
    NvBufSurface* surface = nullptr;
    if (NvBufSurfaceFromFd(slot->rgbaDmabuf, reinterpret_cast<void**>(&surface)) != 0 || !surface) {
        RCLCPP_ERROR(get_logger(), "从 RGBA dmabuf 获取 NvBufSurface 失败");
        NvBufSurf::NvDestroy(slot->rgbaDmabuf);
        slot->rgbaDmabuf = -1;
        return false;
    }
    if (NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_WRITE) != 0) {
        RCLCPP_ERROR(get_logger(), "映射 RGBA dmabuf 失败");
        NvBufSurf::NvDestroy(slot->rgbaDmabuf);
        slot->rgbaDmabuf = -1;
        return false;
    }
    auto* mapped = static_cast<uint32_t*>(surface->surfaceList[0].mappedAddr.addr[0]);
    if (!mapped) {
        NvBufSurfaceUnMap(surface, 0, 0);
        RCLCPP_ERROR(get_logger(), "获取 RGBA dmabuf CPU 地址失败");
        NvBufSurf::NvDestroy(slot->rgbaDmabuf);
        slot->rgbaDmabuf = -1;
        return false;
    }
    constexpr uint32_t kLetterboxRgba = 0xFF727272U;
    const size_t pixelsPerRow = surface->surfaceList[0].pitch / sizeof(uint32_t);
    for (uint32_t row = 0; row < static_cast<uint32_t>(inputSize_); ++row) {
        std::fill_n(mapped + static_cast<size_t>(row) * pixelsPerRow,
                    static_cast<size_t>(inputSize_), kLetterboxRgba);
    }
    const bool synchronized = NvBufSurfaceSyncForDevice(surface, 0, 0) == 0;
    NvBufSurfaceUnMap(surface, 0, 0);
    if (!synchronized) {
        RCLCPP_ERROR(get_logger(), "同步 RGBA dmabuf 到设备失败");
        NvBufSurf::NvDestroy(slot->rgbaDmabuf);
        slot->rgbaDmabuf = -1;
        return false;
    }
    if (!initializeSlotCudaInterop(slot)) {
        NvBufSurf::NvDestroy(slot->rgbaDmabuf);
        slot->rgbaDmabuf = -1;
        return false;
    }
    return true;
}

bool InferenceNode::initializeSlotCudaInterop(StagingSlot* slot) {
    if (!slot || slot->rgbaDmabuf < 0) return false;
    NvBufSurface* surface = nullptr;
    if (NvBufSurfaceFromFd(slot->rgbaDmabuf, reinterpret_cast<void**>(&surface)) != 0 || !surface) {
        RCLCPP_ERROR(get_logger(), "从 RGBA dmabuf 获取 NvBufSurface 失败");
        return false;
    }
    if (NvBufSurfaceMapEglImage(surface, 0) != 0) {
        RCLCPP_ERROR(get_logger(), "创建 RGBA EGL image 失败");
        return false;
    }
    slot->mappedRgbaSurface = surface;
    const auto eglImage = static_cast<EGLImageKHR>(surface->surfaceList[0].mappedAddr.eglImage);
    CUeglFrame eglFrame{};
    if (cudaFree(0) != cudaSuccess || !eglImage) {
        RCLCPP_ERROR(get_logger(), "初始化 CUDA 或取得 RGBA EGL image 失败");
        releaseSlotCudaInterop(slot);
        return false;
    }
    const CUresult registerResult = cuGraphicsEGLRegisterImage(
        &slot->mappedRgbaResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (registerResult != CUDA_SUCCESS) {
        RCLCPP_ERROR(get_logger(), "注册 RGBA EGL image 到 CUDA 失败 (CUresult=%d)",
                     static_cast<int>(registerResult));
        releaseSlotCudaInterop(slot);
        return false;
    }
    const CUresult frameResult = cuGraphicsResourceGetMappedEglFrame(
        &eglFrame, slot->mappedRgbaResource, 0, 0);
    if (frameResult != CUDA_SUCCESS || eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH ||
        !eglFrame.frame.pPitch[0]) {
        RCLCPP_ERROR(get_logger(), "获取 RGBA CUDA EGL frame 失败 (CUresult=%d, frameType=%d)",
                     static_cast<int>(frameResult), static_cast<int>(eglFrame.frameType));
        releaseSlotCudaInterop(slot);
        return false;
    }
    slot->rgbaDevice = eglFrame.frame.pPitch[0];
    slot->rgbaPitch = eglFrame.pitch;
    return true;
}

bool InferenceNode::copyYuvToRgbaGpu(StagingSlot* slot, void** rgbaDevice, size_t* sourcePitch) {
    if (!slot || !rgbaDevice || !sourcePitch || !slot->sourceFrame || slot->rgbaDmabuf < 0 ||
        !slot->rgbaDevice || slot->rgbaPitch == 0) return false;

    NvBufSurf::NvCommonTransformParams transform{};
    transform.src_width = slot->width;
    transform.src_height = slot->height;
    const float scale = std::min(static_cast<float>(inputSize_) / slot->width,
                                 static_cast<float>(inputSize_) / slot->height);
    transform.dst_width = std::max(1U, static_cast<uint32_t>(std::round(slot->width * scale)));
    transform.dst_height = std::max(1U, static_cast<uint32_t>(std::round(slot->height * scale)));
    transform.dst_left = (static_cast<uint32_t>(inputSize_) - transform.dst_width) / 2;
    transform.dst_top = (static_cast<uint32_t>(inputSize_) - transform.dst_height) / 2;
    transform.flag = static_cast<NvBufSurfTransform_Transform_Flag>(
        NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_DST);
    transform.flip = NvBufSurfTransform_None;
    transform.filter = NvBufSurfTransformInter_Bilinear;
    const int sourceDmabuf = slot->sourceFrame->dmabuf();
    NvBufSurface* sourceSurface = nullptr;
    if (NvBufSurfaceFromFd(sourceDmabuf, reinterpret_cast<void**>(&sourceSurface)) != 0 || !sourceSurface) {
        RCLCPP_ERROR(get_logger(), "从采集 dmabuf 获取 NvBufSurface 失败 (fd=%d)", sourceDmabuf);
        return false;
    }
    const int transformResult = NvBufSurf::NvTransform(&transform, sourceDmabuf, slot->rgbaDmabuf);
    if (transformResult != 0) {
        const auto& source = sourceSurface->surfaceList[0];
        RCLCPP_ERROR(
            get_logger(),
            "YUV 转 RGBA 失败 (status=%d, src fd=%d %ux%u format=%d layout=%d planes=%u, dst fd=%d)",
            transformResult, sourceDmabuf, source.width, source.height,
            static_cast<int>(source.colorFormat), static_cast<int>(source.layout),
            source.planeParams.num_planes, slot->rgbaDmabuf);
        return false;
    }
    slot->sourceFrame.reset();
    *rgbaDevice = slot->rgbaDevice;
    *sourcePitch = slot->rgbaPitch;
    return true;
}

void InferenceNode::releaseSlotCudaInterop(StagingSlot* slot) {
    if (!slot) return;
    if (slot->mappedRgbaResource) {
        cuGraphicsUnregisterResource(slot->mappedRgbaResource);
        slot->mappedRgbaResource = nullptr;
    }
    if (slot->mappedRgbaSurface) {
        NvBufSurfaceUnMapEglImage(static_cast<NvBufSurface*>(slot->mappedRgbaSurface), 0);
        slot->mappedRgbaSurface = nullptr;
    }
    slot->rgbaDevice = nullptr;
    slot->rgbaPitch = 0;
}

void InferenceNode::releaseSlot(StagingSlot* slot) {
    if (!slot) return;
    releaseSlotCudaInterop(slot);
    if (slot->rgbaDmabuf >= 0) NvBufSurf::NvDestroy(slot->rgbaDmabuf);
    slot->rgbaDmabuf = -1;
    slot->width = 0;
    slot->height = 0;
    slot->sourceFrame.reset();
}

void InferenceNode::stageFrame(argus_transport::ArgusFramePacket::ConstSharedPtr packet) {
    if (!packet || !packet->frame) {
        RCLCPP_WARN(get_logger(), "收到不含 native frame 的 Argus YUV packet");
        return;
    }
    PendingFrame frame;
    frame.header = packet->header;
    frame.frameNumber = packet->frame_number;
    frame.width = packet->width;
    frame.height = packet->height;
    frame.sourceFrame = packet->frame;

    std::optional<PendingFrame> supersededFrame;
    {
        std::lock_guard<std::mutex> lock(jobsMutex_);
        if (stopInference_) return;
        if (pendingFrame_) {
            supersededFrame = std::move(*pendingFrame_);
            ++supersededFrames_;
        }
        pendingFrame_ = std::move(frame);
    }
    jobsReady_.notify_one();
}

void InferenceNode::inferenceLoop() {
    while (true) {
        PendingFrame frame;
        {
            std::unique_lock<std::mutex> lock(jobsMutex_);
            jobsReady_.wait(lock, [this] { return stopInference_ || pendingFrame_.has_value(); });
            if (stopInference_) return;
            frame = std::move(*pendingFrame_);
            pendingFrame_.reset();
        }
        inferFrame(std::move(frame));
    }
}

// 发布示例分割可视化，十分耗时
bool InferenceNode::publishOverlay(
    const StagingSlot& slot, const std::vector<SegmentationInstance>& instances) {
    NvBufSurface* surface = nullptr;
    if (slot.rgbaDmabuf < 0 || slot.width == 0 || slot.height == 0 || NvBufSurfaceFromFd(
            slot.rgbaDmabuf, reinterpret_cast<void**>(&surface)) != 0 || !surface) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "无法获取推理 RGBA surface，跳过 overlay 发布");
        return false;
    }
    if (NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_READ) != 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "无法将推理 RGBA surface 映射到 CPU，跳过 overlay 发布");
        return false;
    }
    if (NvBufSurfaceSyncForCpu(surface, 0, 0) != 0) {
        NvBufSurfaceUnMap(surface, 0, 0);
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "无法同步推理 RGBA surface 到 CPU，跳过 overlay 发布");
        return false;
    }

    const auto& surfaceParams = surface->surfaceList[0];
    const auto* mapped = static_cast<const uint8_t*>(surfaceParams.mappedAddr.addr[0]);
    if (!mapped || surfaceParams.width == 0 || surfaceParams.height == 0) {
        NvBufSurfaceUnMap(surface, 0, 0);
        return false;
    }
    cv::Mat rgba(static_cast<int>(surfaceParams.height), static_cast<int>(surfaceParams.width),
                 CV_8UC4, const_cast<uint8_t*>(mapped), surfaceParams.pitch);
    cv::Mat overlay;
    cv::cvtColor(rgba, overlay, cv::COLOR_RGBA2BGR);

    const float scale = std::min(
        static_cast<float>(surfaceParams.width) / static_cast<float>(slot.width),
        static_cast<float>(surfaceParams.height) / static_cast<float>(slot.height));
    const int offsetX = (static_cast<int>(surfaceParams.width) -
                         static_cast<int>(std::round(slot.width * scale))) / 2;
    const int offsetY = (static_cast<int>(surfaceParams.height) -
                         static_cast<int>(std::round(slot.height * scale))) / 2;
    for (const auto& instance : instances) {
        const int x = std::clamp(static_cast<int>(std::round(instance.box.x * scale)) + offsetX,
                                 0, overlay.cols - 1);
        const int y = std::clamp(static_cast<int>(std::round(instance.box.y * scale)) + offsetY,
                                 0, overlay.rows - 1);
        const int width = std::clamp(static_cast<int>(std::round(instance.box.width * scale)),
                                     1, overlay.cols - x);
        const int height = std::clamp(static_cast<int>(std::round(instance.box.height * scale)),
                                      1, overlay.rows - y);
        const cv::Scalar color(
            80.0 + (instance.classId * 67) % 150,
            80.0 + (instance.classId * 41) % 150,
            80.0 + (instance.classId * 97) % 150);
        cv::Mat resizedMask;
        if (!instance.mask.empty()) {
            cv::resize(instance.mask, resizedMask, cv::Size(width, height), 0.0, 0.0,
                       cv::INTER_NEAREST);
            cv::Mat region = overlay(cv::Rect(x, y, width, height));
            for (int row = 0; row < region.rows; ++row) {
                for (int column = 0; column < region.cols; ++column) {
                    if (resizedMask.at<uint8_t>(row, column) != 0) {
                        auto& pixel = region.at<cv::Vec3b>(row, column);
                        for (int channel = 0; channel < 3; ++channel) {
                            pixel[channel] = static_cast<uint8_t>(
                                pixel[channel] * 0.55 + color[channel] * 0.45);
                        }
                    }
                }
            }
        }
        cv::rectangle(overlay, cv::Rect(x, y, width, height), color, 2);
        const std::string label = instance.className + " " +
                                  std::to_string(static_cast<int>(instance.confidence * 100.0F)) + "%";
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        const int textTop = std::max(0, y - textSize.height - baseline - 4);
        cv::rectangle(overlay, cv::Rect(x, textTop, std::min(textSize.width + 8, overlay.cols - x),
                                       textSize.height + baseline + 4), color, cv::FILLED);
        cv::putText(overlay, label, cv::Point(x + 4, textTop + textSize.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    std::vector<uint8_t> encoded;
    const std::vector<int> encodeParams{cv::IMWRITE_JPEG_QUALITY, overlayQuality_};
    const bool encodedOk = cv::imencode(".jpg", overlay, encoded, encodeParams);
    NvBufSurfaceUnMap(surface, 0, 0);
    if (!encodedOk) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "overlay JPEG 编码失败");
        return false;
    }
    sensor_msgs::msg::CompressedImage message;
    message.header = slot.header;
    message.format = "jpeg";
    message.data = std::move(encoded);
    publisher_->publish(std::move(message));
    return true;
}

void InferenceNode::inferFrame(PendingFrame frame) {
    void* rgbaDevice = nullptr;
    size_t sourcePitch = 0;
    auto& slot = inferenceSlot_;
    slot.header = std::move(frame.header);
    slot.frameNumber = frame.frameNumber;
    slot.width = frame.width;
    slot.height = frame.height;
    slot.sourceFrame = std::move(frame.sourceFrame);
    if (!copyYuvToRgbaGpu(&slot, &rgbaDevice, &sourcePitch)) {
        slot.sourceFrame.reset();
        return;
    }
    std::vector<SegmentationInstance> instances;
    InferenceTiming modelTiming;
    const bool inferred = model_->infer(rgbaDevice, sourcePitch, slot.width, slot.height, inputSize_, inputSize_,
                                        confidenceThreshold_, iouThreshold_, &instances, &modelTiming);
    if (!inferred) {
        if (!model_->synchronize()) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                                  "TensorRT 推理失败后等待 CUDA stream 失败");
        }
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "YOLOv8-seg TensorRT 推理失败");
        return;
    }

    if (enableOverlay_) {
        publishOverlay(slot, instances);
    }
    ++processedFrames_;
    if (processedFrames_ % timingLogEveryNFrames_ == 0) {
        RCLCPP_INFO(
            get_logger(),
            "模型性能 #%lu（源帧 #%lu，%ux%u，%zu 个实例）：GPU 预处理 %.2f ms，TensorRT %.2f ms，输出回传 %.2f ms，候选框解码 %.2f ms，NMS %.2f ms，掩码解码 %.2f ms，模型端到端 %.2f ms；覆盖等待帧 %lu",
            static_cast<unsigned long>(processedFrames_), static_cast<unsigned long>(slot.frameNumber),
            slot.width, slot.height, instances.size(), modelTiming.gpuPreprocessMs, modelTiming.tensorRtMs,
            modelTiming.outputCopyMs, modelTiming.candidateDecodeMs, modelTiming.nmsMs,
            modelTiming.maskDecodeMs, modelTiming.totalMs,
            static_cast<unsigned long>(supersededFrames_.load()));
    }
}

}  // namespace argus_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(argus_inference::InferenceNode)
