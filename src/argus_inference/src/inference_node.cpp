#include "argus_inference/inference_node.hpp"

#include "argus_inference/yolov8_segmentation.hpp"

#include <EGLStream/Frame.h>
#include <EGLStream/Image.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <NvBufSurface.h>
#include <nvbufsurface.h>

#include <EGL/egl.h>
#include <cudaEGL.h>

#include <chrono>
#include <stdexcept>
#include <utility>

namespace argus_inference {

InferenceNode::InferenceNode(const rclcpp::NodeOptions& options)
    : Node("yuv_inference_node", options) {
    inputTopic_ = declare_parameter<std::string>("input_topic", "/camera/image/yuv");
    const auto outputTopic = declare_parameter<std::string>("output_topic", "/camera/inference/segmentation");
    const auto enginePath = declare_parameter<std::string>("engine_path", "/home/royfan/yolov8_trt/yolov8s-seg-640.engine");
    const auto inputSize = declare_parameter<int>("input_size", 640);
    const auto requireFp16Engine = declare_parameter<bool>("require_fp16_engine", true);
    const auto stagingBufferCount = declare_parameter<int>("staging_buffer_count", 3);
    confidenceThreshold_ = static_cast<float>(declare_parameter<double>("confidence_threshold", 0.25));
    iouThreshold_ = static_cast<float>(declare_parameter<double>("iou_threshold", 0.45));
    if (inputSize <= 0 || stagingBufferCount < 2 || confidenceThreshold_ < 0.0F || confidenceThreshold_ > 1.0F ||
        iouThreshold_ < 0.0F || iouThreshold_ > 1.0F) {
        throw std::invalid_argument("TensorRT inference parameter is invalid");
    }

    model_ = std::make_unique<YoloV8Segmentation>();
    if (!model_->initialize(enginePath, inputSize, requireFp16Engine)) {
        throw std::runtime_error("无法加载 YOLOv8 TensorRT engine: " + enginePath);
    }
    frameSlots_.resize(static_cast<size_t>(stagingBufferCount));
    publisher_ = create_publisher<argus_interfaces::msg::ArgusInferenceResult>(
        outputTopic, rclcpp::QoS(rclcpp::KeepLast(8)).reliable());
    inferenceThread_ = std::thread(&InferenceNode::inferenceLoop, this);
    subscription_ = create_subscription<argus_transport::ArgusFramePacket>(
        inputTopic_, rclcpp::QoS(rclcpp::KeepLast(8)).reliable(),
        [this](argus_transport::ArgusFramePacket::ConstSharedPtr message) { stageFrame(std::move(message)); });
}

InferenceNode::~InferenceNode() {
    subscription_.reset();
    {
        std::lock_guard<std::mutex> lock(jobsMutex_);
        stopInference_ = true;
    }
    jobsReady_.notify_one();
    if (inferenceThread_.joinable()) inferenceThread_.join();
    for (auto& slot : frameSlots_) releaseSlot(&slot);
}

bool InferenceNode::ensureSlotSurfaces(FrameSlot* slot, uint32_t width, uint32_t height) {
    if (!slot) return false;
    if (width == slot->width && height == slot->height && slot->rgbaDmabuf >= 0) return true;
    releaseSlotCudaInterop(slot);
    if (slot->rgbaDmabuf >= 0) NvBufSurf::NvDestroy(slot->rgbaDmabuf);
    slot->rgbaDmabuf = -1;
    NvBufSurf::NvCommonAllocateParams params{};
    params.width = width;
    params.height = height;
    params.memType = NVBUF_MEM_SURFACE_ARRAY;
    params.memtag = NvBufSurfaceTag_NONE;
    params.layout = NVBUF_LAYOUT_PITCH;
    params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    if (NvBufSurf::NvAllocate(&params, 1, &slot->rgbaDmabuf) < 0) {
        RCLCPP_ERROR(get_logger(), "创建 RGBA 推理 dmabuf 失败");
        slot->rgbaDmabuf = -1;
        return false;
    }
    slot->width = width;
    slot->height = height;
    return true;
}

bool InferenceNode::copyFrameToYuvBuffer(const argus_transport::ArgusFramePacket& packet, FrameSlot* slot) {
    if (!packet.frame || !slot || packet.width == 0 || packet.height == 0) return false;
    auto* iFrame = Argus::interface_cast<EGLStream::IFrame>(static_cast<Argus::InterfaceProvider*>(packet.frame->get()));
    auto* image = iFrame ? iFrame->getImage() : nullptr;
    auto* nativeBuffer = image ? Argus::interface_cast<EGLStream::NV::IImageNativeBuffer>(image) : nullptr;
    if (!nativeBuffer) {
        RCLCPP_ERROR(get_logger(), "Argus image 不支持 native buffer");
        return false;
    }
    if (packet.width != slot->width || packet.height != slot->height || slot->yuvDmabuf < 0) {
        releaseSlot(slot);
        slot->yuvDmabuf = nativeBuffer->createNvBuffer(Argus::Size2D<uint32_t>(packet.width, packet.height),
                                                        NVBUF_COLOR_FORMAT_YUV420, NVBUF_LAYOUT_PITCH);
        if (slot->yuvDmabuf < 0) {
            RCLCPP_ERROR(get_logger(), "创建 YUV 推理 dmabuf 失败");
            return false;
        }
        if (!ensureSlotSurfaces(slot, packet.width, packet.height)) {
            releaseSlot(slot);
            return false;
        }
    } else if (nativeBuffer->copyToNvBuffer(slot->yuvDmabuf) != Argus::STATUS_OK) {
        RCLCPP_ERROR(get_logger(), "复制 Argus image 到推理 dmabuf 失败");
        return false;
    }
    return true;
}

bool InferenceNode::copyYuvToRgbaGpu(FrameSlot* slot, void** rgbaDevice, size_t* sourcePitch) {
    if (!slot || !rgbaDevice || !sourcePitch || slot->yuvDmabuf < 0 || slot->rgbaDmabuf < 0) return false;

    NvBufSurf::NvCommonTransformParams transform{};
    transform.src_width = slot->width;
    transform.src_height = slot->height;
    transform.dst_width = slot->width;
    transform.dst_height = slot->height;
    transform.flag = NVBUFSURF_TRANSFORM_FILTER;
    transform.flip = NvBufSurfTransform_None;
    transform.filter = NvBufSurfTransformInter_Algo3;
    if (NvBufSurf::NvTransform(&transform, slot->yuvDmabuf, slot->rgbaDmabuf) != 0) {
        RCLCPP_ERROR(get_logger(), "YUV 转 RGBA 失败");
        return false;
    }
    NvBufSurface* surface = nullptr;
    if (NvBufSurfaceFromFd(slot->rgbaDmabuf, reinterpret_cast<void**>(&surface)) != 0 || !surface) {
        RCLCPP_ERROR(get_logger(), "从 RGBA dmabuf 获取 NvBufSurface 失败");
        return false;
    }
    if (NvBufSurfaceMapEglImage(surface, 0) != 0) {
        RCLCPP_ERROR(get_logger(), "创建 RGBA EGL image 失败");
        return false;
    }
    const auto& item = surface->surfaceList[0];
    const auto eglImage = static_cast<EGLImageKHR>(item.mappedAddr.eglImage);
    CUeglFrame eglFrame{};
    if (cudaFree(0) != cudaSuccess || !eglImage) {
        RCLCPP_ERROR(get_logger(), "初始化 CUDA 或取得 RGBA EGL image 失败");
        NvBufSurfaceUnMapEglImage(surface, 0);
        return false;
    }
    const CUresult registerResult = cuGraphicsEGLRegisterImage(
        &slot->mappedRgbaResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (registerResult != CUDA_SUCCESS) {
        RCLCPP_ERROR(get_logger(), "注册 RGBA EGL image 到 CUDA 失败 (CUresult=%d)",
                     static_cast<int>(registerResult));
        NvBufSurfaceUnMapEglImage(surface, 0);
        return false;
    }
    const CUresult frameResult = cuGraphicsResourceGetMappedEglFrame(&eglFrame, slot->mappedRgbaResource, 0, 0);
    if (frameResult != CUDA_SUCCESS || eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH ||
        !eglFrame.frame.pPitch[0]) {
        RCLCPP_ERROR(get_logger(), "获取 RGBA CUDA EGL frame 失败 (CUresult=%d, frameType=%d)",
                     static_cast<int>(frameResult), static_cast<int>(eglFrame.frameType));
        cuGraphicsUnregisterResource(slot->mappedRgbaResource);
        slot->mappedRgbaResource = nullptr;
        NvBufSurfaceUnMapEglImage(surface, 0);
        return false;
    }
    slot->mappedRgbaSurface = surface;
    *rgbaDevice = eglFrame.frame.pPitch[0];
    *sourcePitch = eglFrame.pitch;
    return true;
}

void InferenceNode::releaseSlotCudaInterop(FrameSlot* slot) {
    if (!slot) return;
    if (slot->mappedRgbaResource) {
        cuGraphicsUnregisterResource(slot->mappedRgbaResource);
        slot->mappedRgbaResource = nullptr;
    }
    if (slot->mappedRgbaSurface) {
        NvBufSurfaceUnMapEglImage(static_cast<NvBufSurface*>(slot->mappedRgbaSurface), 0);
        slot->mappedRgbaSurface = nullptr;
    }
}

void InferenceNode::releaseSlot(FrameSlot* slot) {
    if (!slot) return;
    releaseSlotCudaInterop(slot);
    if (slot->yuvDmabuf >= 0) NvBufSurf::NvDestroy(slot->yuvDmabuf);
    if (slot->rgbaDmabuf >= 0) NvBufSurf::NvDestroy(slot->rgbaDmabuf);
    slot->yuvDmabuf = -1;
    slot->rgbaDmabuf = -1;
    slot->width = 0;
    slot->height = 0;
}

bool InferenceNode::reserveSlot(size_t* slotIndex) {
    if (!slotIndex) return false;
    std::lock_guard<std::mutex> lock(jobsMutex_);
    for (size_t index = 0; index < frameSlots_.size(); ++index) {
        if (!frameSlots_[index].inUse) {
            frameSlots_[index].inUse = true;
            *slotIndex = index;
            return true;
        }
    }
    if (pendingJobs_.empty()) return false;
    const size_t supersededSlot = pendingJobs_.front().slotIndex;
    pendingJobs_.pop_front();
    frameSlots_[supersededSlot].inUse = true;
    *slotIndex = supersededSlot;
    ++supersededFrames_;
    return true;
}

void InferenceNode::releaseJobSlot(size_t slotIndex) {
    std::lock_guard<std::mutex> lock(jobsMutex_);
    frameSlots_[slotIndex].inUse = false;
}

void InferenceNode::stageFrame(argus_transport::ArgusFramePacket::ConstSharedPtr packet) {
    if (!packet || !packet->frame) {
        RCLCPP_WARN(get_logger(), "收到不含 native frame 的 Argus YUV packet");
        return;
    }
    size_t slotIndex = 0;
    if (!reserveSlot(&slotIndex)) {
        ++droppedFrames_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "推理槽位全部在执行，已丢弃源帧 #%lu",
                             static_cast<unsigned long>(packet->frame_number));
        return;
    }
    const auto stagingStart = std::chrono::steady_clock::now();
    if (!copyFrameToYuvBuffer(*packet, &frameSlots_[slotIndex])) {
        releaseJobSlot(slotIndex);
        return;
    }
    FrameJob job;
    job.header = packet->header;
    job.frameNumber = packet->frame_number;
    job.width = packet->width;
    job.height = packet->height;
    job.slotIndex = slotIndex;
    job.stagingMs = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - stagingStart).count());
    job.enqueuedAt = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(jobsMutex_);
        if (stopInference_) {
            frameSlots_[slotIndex].inUse = false;
            return;
        }
        pendingJobs_.push_back(std::move(job));
    }
    jobsReady_.notify_one();
}

void InferenceNode::inferenceLoop() {
    while (true) {
        FrameJob job;
        {
            std::unique_lock<std::mutex> lock(jobsMutex_);
            jobsReady_.wait(lock, [this] { return stopInference_ || !pendingJobs_.empty(); });
            if (stopInference_ && pendingJobs_.empty()) return;
            job = std::move(pendingJobs_.front());
            pendingJobs_.pop_front();
        }
        inferFrame(job);
        releaseJobSlot(job.slotIndex);
    }
}

void InferenceNode::inferFrame(const FrameJob& job) {
    const auto processingStart = std::chrono::steady_clock::now();
    void* rgbaDevice = nullptr;
    size_t sourcePitch = 0;
    auto& slot = frameSlots_[job.slotIndex];
    if (!copyYuvToRgbaGpu(&slot, &rgbaDevice, &sourcePitch)) return;
    std::vector<SegmentationInstance> instances;
    const auto inferenceStart = std::chrono::steady_clock::now();
    const bool inferred = model_->infer(rgbaDevice, sourcePitch, job.width, job.height,
                                        confidenceThreshold_, iouThreshold_, &instances);
    releaseSlotCudaInterop(&slot);
    if (!inferred) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "YOLOv8-seg TensorRT 推理失败");
        return;
    }
    const float inferenceMs = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inferenceStart).count());

    argus_interfaces::msg::ArgusInferenceResult result;
    result.header = job.header;
    result.frame_number = job.frameNumber;
    result.image_width = job.width;
    result.image_height = job.height;
    result.inference_ms = inferenceMs;
    result.instances.reserve(instances.size());
    for (const auto& instance : instances) {
        argus_interfaces::msg::ArgusInstanceSegmentation message;
        message.class_id = instance.classId;
        message.class_name = instance.className;
        message.confidence = instance.confidence;
        message.x_min = static_cast<float>(instance.box.x);
        message.y_min = static_cast<float>(instance.box.y);
        message.x_max = static_cast<float>(instance.box.br().x);
        message.y_max = static_cast<float>(instance.box.br().y);
        message.mask_x = static_cast<uint32_t>(instance.box.x);
        message.mask_y = static_cast<uint32_t>(instance.box.y);
        message.mask_width = static_cast<uint32_t>(instance.mask.cols);
        message.mask_height = static_cast<uint32_t>(instance.mask.rows);
        for (int row = 0; row < instance.mask.rows; ++row) {
            const auto* data = instance.mask.ptr<uint8_t>(row);
            message.mask.insert(message.mask.end(), data, data + instance.mask.cols);
        }
        result.instances.push_back(std::move(message));
    }
    publisher_->publish(std::move(result));
    ++processedFrames_;
    const float processingMs = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - processingStart).count());
    const float queueMs = static_cast<float>(std::chrono::duration<double, std::milli>(
        processingStart - job.enqueuedAt).count());
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                         "已发布推理结果 #%lu（源帧 #%lu，%zu 个实例；暂存 %.1f ms，排队 %.1f ms，推理 %.1f ms，处理 %.1f ms；丢弃 %lu，替换旧帧 %lu）",
                         static_cast<unsigned long>(processedFrames_),
                         static_cast<unsigned long>(job.frameNumber), instances.size(), job.stagingMs,
                         queueMs, inferenceMs, processingMs,
                         static_cast<unsigned long>(droppedFrames_.load()),
                         static_cast<unsigned long>(supersededFrames_.load()));
}

}  // namespace argus_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(argus_inference::InferenceNode)
