#include "argus_camera/argus_camera_node.hpp"

#include "argus_camera/camera_controls.hpp"

#include <Argus/Argus.h>
#include <Argus/CaptureMetadata.h>
#include <NvBufSurface.h>

#include <EGL/egl.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace argus_camera {
namespace {

constexpr uint64_t kFrameTimeoutNs = 5ULL * 1000ULL * 1000ULL * 1000ULL;

bool parseDenoiseMode(const std::string& value, Argus::DenoiseMode* mode) {
    if (value == "off") *mode = Argus::DENOISE_MODE_OFF;
    else if (value == "fast") *mode = Argus::DENOISE_MODE_FAST;
    else if (value == "hq") *mode = Argus::DENOISE_MODE_HIGH_QUALITY;
    else return false;
    return true;
}

bool parseEdgeEnhanceMode(const std::string& value, Argus::EdgeEnhanceMode* mode) {
    if (value == "off") *mode = Argus::EDGE_ENHANCE_MODE_OFF;
    else if (value == "fast") *mode = Argus::EDGE_ENHANCE_MODE_FAST;
    else if (value == "hq") *mode = Argus::EDGE_ENHANCE_MODE_HIGH_QUALITY;
    else return false;
    return true;
}
}  // namespace

ArgusCameraNode::ArgusCameraNode(const rclcpp::NodeOptions& options)
    : Node("argus_camera_node", options) {
    const auto topic = declare_parameter<std::string>("topic", "/camera/image/yuv");
    frameId_ = declare_parameter<std::string>("frame_id", "camera");
    requestedFrames_ = declare_parameter<int64_t>("frame_count", 0);
    cameraIndex_ = declare_parameter<int>("camera_index", 0);
    sensorModeIndex_ = declare_parameter<int>("sensor_mode_index", 0);
    fifoLength_ = declare_parameter<int>("fifo_length", 8);
    captureBufferCount_ = declare_parameter<int>("capture_buffer_count", fifoLength_);
    frameRate_ = declare_parameter<double>("frame_rate", 0.0);
    controls_.frameRate = frameRate_;
    controls_.saturation = static_cast<float>(declare_parameter<double>("saturation", 1.0));
    controls_.exposureCompensation = static_cast<float>(
        declare_parameter<double>("exposure_compensation", 0.0));
    controls_.ispDigitalGain = static_cast<float>(
        declare_parameter<double>("isp_digital_gain", 1.0));
    controls_.denoiseStrength = static_cast<float>(
        declare_parameter<double>("denoise_strength", 1.0));
    controls_.edgeStrength = static_cast<float>(
        declare_parameter<double>("edge_enhance_strength", 1.0));
    controls_.manualWb = declare_parameter<bool>("manual_white_balance", false);
    const auto denoiseMode = declare_parameter<std::string>("denoise_mode", "fast");
    const auto edgeEnhanceMode = declare_parameter<std::string>("edge_enhance_mode", "fast");
    const auto whiteBalanceGains = declare_parameter<std::vector<double>>(
        "white_balance_gains", {1.0, 1.0, 1.0, 1.0});

    if (requestedFrames_ < 0 || cameraIndex_ < 0 || sensorModeIndex_ < 0 || fifoLength_ <= 0 ||
        captureBufferCount_ < 2 ||
        !std::isfinite(frameRate_) || frameRate_ < 0.0 || frameRate_ > 1.0e9 ||
        controls_.saturation < 0.0f || controls_.saturation > 2.0f ||
        controls_.ispDigitalGain <= 0.0f || controls_.denoiseStrength < 0.0f ||
        controls_.denoiseStrength > 1.0f || controls_.edgeStrength < 0.0f ||
        controls_.edgeStrength > 1.0f || whiteBalanceGains.size() != 4 ||
        !parseDenoiseMode(denoiseMode, &controls_.denoiseMode) ||
        !parseEdgeEnhanceMode(edgeEnhanceMode, &controls_.edgeMode)) {
        throw std::invalid_argument("Argus camera parameter is invalid");
    }
    for (const auto gain : whiteBalanceGains) {
        if (gain <= 0.0) throw std::invalid_argument("white_balance_gains must be positive");
    }
    controls_.wbGains = Argus::BayerTuple<float>(
        static_cast<float>(whiteBalanceGains[0]), static_cast<float>(whiteBalanceGains[1]),
        static_cast<float>(whiteBalanceGains[2]), static_cast<float>(whiteBalanceGains[3]));

    publisher_ = create_publisher<argus_transport::ArgusFramePacket>(
        topic, rclcpp::QoS(rclcpp::KeepLast(8)).reliable());
    if (!start()) throw std::runtime_error("Argus camera start failed");
}

ArgusCameraNode::~ArgusCameraNode() {
    stop();
    destroyCaptureBufferPool();
}

bool ArgusCameraNode::start() {
    if (started_.exchange(true)) return true;
    provider_.reset(Argus::CameraProvider::create());
    iProvider_ = Argus::interface_cast<Argus::ICameraProvider>(provider_);
    if (!continuous_capture::check(iProvider_ != nullptr,
                                   "无法获取 ICameraProvider，请检查 nvargus-daemon")) return false;
    std::vector<Argus::CameraDevice*> devices;
    if (!continuous_capture::ok(iProvider_->getCameraDevices(&devices), "枚举摄像头失败") ||
        !continuous_capture::check(cameraIndex_ < static_cast<int>(devices.size()),
                                   "camera_index 超出可用摄像头范围")) return false;
    auto* device = devices[cameraIndex_];
    auto* properties = Argus::interface_cast<Argus::ICameraProperties>(device);
    std::vector<Argus::SensorMode*> modes;
    if (!continuous_capture::check(properties != nullptr, "无法获取相机属性") ||
        !continuous_capture::ok(properties->getAllSensorModes(&modes), "枚举 sensor mode 失败") ||
        !continuous_capture::check(sensorModeIndex_ < static_cast<int>(modes.size()),
                                   "sensor_mode_index 超出可用 sensor mode 范围")) return false;
    for (size_t index = 0; index < modes.size(); ++index) {
        auto* modeInterface = Argus::interface_cast<Argus::ISensorMode>(modes[index]);
        if (!modeInterface) continue;
        const auto modeResolution = modeInterface->getResolution();
        RCLCPP_INFO(get_logger(), "sensor_mode_index=%zu：%ux%u", index,
                    modeResolution.width(), modeResolution.height());
    }
    sensorMode_ = modes[sensorModeIndex_];
    auto* sensorModeInterface = Argus::interface_cast<Argus::ISensorMode>(sensorMode_);
    if (!continuous_capture::check(sensorModeInterface != nullptr, "无法获取 sensor mode 接口")) return false;
    resolution_ = sensorModeInterface->getResolution();
    const auto frameDurationRange = sensorModeInterface->getFrameDurationRange();
    RCLCPP_INFO(get_logger(),
                "已选择 sensor_mode_index=%d，YUV 输出分辨率=%ux%u，支持帧周期=%llu..%llu ns",
                sensorModeIndex_, resolution_.width(), resolution_.height(),
                static_cast<unsigned long long>(frameDurationRange.min()),
                static_cast<unsigned long long>(frameDurationRange.max()));
    if (frameRate_ > 0.0) {
        RCLCPP_INFO(get_logger(), "请求采集帧率=%.3f FPS", frameRate_);
    }

    Argus::Status status = Argus::STATUS_OK;
    session_.reset(iProvider_->createCaptureSession(device, &status));
    if (!continuous_capture::ok(status, "创建 CaptureSession 失败") ||
        !continuous_capture::check(static_cast<bool>(session_), "CaptureSession 为空")) return false;
    iSession_ = Argus::interface_cast<Argus::ICaptureSession>(session_);
    settings_.reset(iSession_->createOutputStreamSettings(Argus::STREAM_TYPE_BUFFER));
    auto* streamSettings = Argus::interface_cast<Argus::IBufferOutputStreamSettings>(settings_);
    if (!continuous_capture::check(streamSettings != nullptr, "无法获取 stream settings") ||
        !continuous_capture::ok(streamSettings->setBufferType(Argus::BUFFER_TYPE_EGL_IMAGE),
                                "设置 EGLImage buffer 类型失败")) return false;
    streamSettings->setMetadataEnable(true);
    stream_.reset(iSession_->createOutputStream(settings_.get(), &status));
    if (!continuous_capture::ok(status, "创建 OutputStream 失败") ||
        !continuous_capture::check(static_cast<bool>(stream_), "OutputStream 为空")) return false;
    iBufferStream_ = Argus::interface_cast<Argus::IBufferOutputStream>(stream_);
    if (!continuous_capture::check(iBufferStream_ != nullptr, "创建 BufferOutputStream 失败")) return false;
    bufferReleaseState_ = std::make_shared<argus_transport::ArgusBufferReleaseState>();
    bufferReleaseState_->setStream(iBufferStream_);
    if (!createCaptureBufferPool()) {
        bufferReleaseState_->disable();
        destroyCaptureBufferPool();
        return false;
    }
    captureThread_ = std::thread(&ArgusCameraNode::captureLoop, this);
    return true;
}

void ArgusCameraNode::stop() {
    quit_ = true;
    if (iSession_) iSession_->stopRepeat();
    if (captureThread_.joinable()) captureThread_.join();
    if (iSession_) iSession_->waitForIdle(kFrameTimeoutNs);
    if (iBufferStream_) iBufferStream_->endOfStream();
    if (bufferReleaseState_) bufferReleaseState_->disable();
}

bool ArgusCameraNode::createCaptureBufferPool() {
    eglDisplay_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!continuous_capture::check(eglDisplay_ != EGL_NO_DISPLAY, "获取 EGLDisplay 失败") ||
        !continuous_capture::check(eglInitialize(eglDisplay_, nullptr, nullptr) == EGL_TRUE,
                                   "初始化 EGLDisplay 失败")) {
        eglDisplay_ = EGL_NO_DISPLAY;
        return false;
    }
    eglInitialized_ = true;

    Argus::Status status = Argus::STATUS_OK;
    bufferSettings_.reset(iBufferStream_->createBufferSettings(&status));
    auto* imageSettings = Argus::interface_cast<Argus::IEGLImageBufferSettings>(bufferSettings_);
    if (!continuous_capture::ok(status, "创建 BufferSettings 失败") ||
        !continuous_capture::check(static_cast<bool>(bufferSettings_), "BufferSettings 为空") ||
        !continuous_capture::check(imageSettings != nullptr, "无法获取 EGLImageBufferSettings") ||
        !continuous_capture::ok(imageSettings->setEGLDisplay(eglDisplay_), "设置 EGLDisplay 失败")) {
        destroyCaptureBufferPool();
        return false;
    }

    captureBufferInfos_.resize(static_cast<size_t>(captureBufferCount_));
    captureBuffers_.reserve(static_cast<size_t>(captureBufferCount_));
    captureSurfaces_.reserve(static_cast<size_t>(captureBufferCount_));
    NvBufSurfaceAllocateParams params{};
    params.params.width = resolution_.width();
    params.params.height = resolution_.height();
    params.params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    params.params.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
    params.params.memType = NVBUF_MEM_SURFACE_ARRAY;
    params.memtag = NvBufSurfaceTag_CAMERA;
    for (int index = 0; index < captureBufferCount_; ++index) {
        NvBufSurface* captureSurface = nullptr;
        if (!continuous_capture::check(
                NvBufSurfaceAllocate(&captureSurface, 1, &params) == 0 && captureSurface != nullptr,
                "创建采集 NvBufSurface 失败")) {
            destroyCaptureBufferPool();
            return false;
        }
        captureSurface->numFilled = 1;
        captureSurfaces_.push_back(captureSurface);
        if (!continuous_capture::check(NvBufSurfaceMapEglImage(captureSurface, 0) == 0,
                                       "将采集 NvBufSurface 映射为 EGLImage 失败")) {
            destroyCaptureBufferPool();
            return false;
        }
        const auto& surface = captureSurface->surfaceList[0];
        auto& info = captureBufferInfos_[static_cast<size_t>(index)];
        info.dmabuf = static_cast<int>(surface.bufferDesc);
        info.yStride = surface.planeParams.pitch[0];
        info.uStride = surface.planeParams.num_planes > 1 ? surface.planeParams.pitch[1] : 0;
        info.vStride = surface.planeParams.num_planes > 2 ? surface.planeParams.pitch[2] : 0;
        if (!continuous_capture::ok(
                imageSettings->setEGLImage(static_cast<EGLImageKHR>(surface.mappedAddr.eglImage)),
                "设置采集 EGLImage 失败")) {
            destroyCaptureBufferPool();
            return false;
        }
        captureBuffers_.emplace_back(iBufferStream_->createBuffer(bufferSettings_.get(), &status));
        auto* buffer = captureBuffers_.back().get();
        auto* iBuffer = Argus::interface_cast<Argus::IBuffer>(buffer);
        if (!continuous_capture::ok(status, "创建 Argus buffer 失败") ||
            !continuous_capture::check(buffer != nullptr && iBuffer != nullptr, "Argus buffer 为空")) {
            destroyCaptureBufferPool();
            return false;
        }
        iBuffer->setClientData(&info);
        if (!continuous_capture::ok(iBufferStream_->releaseBuffer(buffer), "将采集 buffer 交给 Argus 失败")) {
            destroyCaptureBufferPool();
            return false;
        }
    }
    RCLCPP_INFO(get_logger(), "已创建 %d 个 %ux%u NV12 block-linear NVMM 采集 buffer",
                captureBufferCount_, resolution_.width(), resolution_.height());
    return true;
}

void ArgusCameraNode::destroyCaptureBufferPool() {
    if (bufferReleaseState_) bufferReleaseState_->disable();
    captureBuffers_.clear();
    bufferSettings_.reset();
    captureBufferInfos_.clear();
    for (auto* surface : captureSurfaces_) {
        NvBufSurfaceUnMapEglImage(surface, 0);
        NvBufSurfaceDestroy(surface);
    }
    captureSurfaces_.clear();
    if (eglInitialized_) {
        eglTerminate(eglDisplay_);
        eglInitialized_ = false;
    }
    eglDisplay_ = EGL_NO_DISPLAY;
}

void ArgusCameraNode::captureLoop() {
    Argus::Status status = Argus::STATUS_OK;
    Argus::UniqueObj<Argus::Request> request;
    bool repeating = false;
    request.reset(iSession_->createRequest(Argus::CAPTURE_INTENT_VIDEO_RECORD, &status));
    if (!continuous_capture::ok(status, "创建 request 失败") ||
        !continuous_capture::check(static_cast<bool>(request), "request 为空") ||
        !continuous_capture::configureRequest(request.get(), sensorMode_, stream_.get(), controls_) ||
        !continuous_capture::ok(iSession_->repeat(request.get()), "启动 repeat capture 失败")) {
        quit_ = true;
        return;
    }
    repeating = true;
    uint64_t publishedFrames = 0;
    while (rclcpp::ok() && !quit_ &&
           (requestedFrames_ <= 0 || publishedFrames < static_cast<uint64_t>(requestedFrames_))) {
        auto* buffer = iBufferStream_->acquireBuffer(kFrameTimeoutNs, &status);
        if (!continuous_capture::ok(status, "获取采集 buffer 失败") ||
            !continuous_capture::check(buffer != nullptr, "采集 buffer 为空")) { quit_ = true; break; }
        auto* iBuffer = Argus::interface_cast<Argus::IBuffer>(buffer);
        const auto* bufferInfo = iBuffer
            ? static_cast<const CaptureBufferInfo*>(iBuffer->getClientData())
            : nullptr;
        auto owner = std::make_shared<argus_transport::ArgusFrameOwner>(
            bufferInfo ? bufferInfo->dmabuf : -1, buffer, bufferReleaseState_);
        const auto* metadata = iBuffer ? iBuffer->getMetadata() : nullptr;
        const auto* captureMetadata = metadata
            ? Argus::interface_cast<const Argus::ICaptureMetadata>(metadata)
            : nullptr;
        if (!continuous_capture::check(bufferInfo != nullptr, "无法读取采集 buffer 信息") ||
            !continuous_capture::check(captureMetadata != nullptr, "无法读取 Argus sensor timestamp")) {
            quit_ = true;
            break;
        }
        const uint64_t sensorTimestampNs = captureMetadata->getSensorTimestamp();
        auto packet = std::make_unique<argus_transport::ArgusFramePacket>();
        packet->header.stamp.sec = static_cast<int32_t>(sensorTimestampNs / 1000000000ULL);
        packet->header.stamp.nanosec = static_cast<uint32_t>(sensorTimestampNs % 1000000000ULL);
        packet->header.frame_id = frameId_;
        packet->frame_number = captureMetadata->getCaptureId();
        packet->width = resolution_.width();
        packet->height = resolution_.height();
        packet->y_stride = bufferInfo->yStride;
        packet->u_stride = bufferInfo->uStride;
        packet->v_stride = bufferInfo->vStride;
        packet->encoding = "NV12";
        packet->frame = std::move(owner);
        publisher_->publish(std::move(packet));
        ++publishedFrames;
    }
    if (repeating) iSession_->stopRepeat();
}

}  // namespace argus_camera

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(argus_camera::ArgusCameraNode)
