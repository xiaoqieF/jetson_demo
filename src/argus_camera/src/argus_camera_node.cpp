#include "argus_camera/argus_camera_node.hpp"

#include "argus_camera/camera_controls.hpp"

#include <Argus/Argus.h>
#include <Argus/CaptureMetadata.h>
#include <EGLStream/ArgusCaptureMetadata.h>
#include <EGLStream/Frame.h>
#include <EGLStream/Image.h>

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

ArgusCameraNode::~ArgusCameraNode() { stop(); }

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
    RCLCPP_INFO(get_logger(), "已选择 sensor_mode_index=%d，YUV 输出分辨率=%ux%u",
                sensorModeIndex_, resolution_.width(), resolution_.height());

    Argus::Status status = Argus::STATUS_OK;
    session_.reset(iProvider_->createCaptureSession(device, &status));
    if (!continuous_capture::ok(status, "创建 CaptureSession 失败") ||
        !continuous_capture::check(static_cast<bool>(session_), "CaptureSession 为空")) return false;
    iSession_ = Argus::interface_cast<Argus::ICaptureSession>(session_);
    settings_.reset(iSession_->createOutputStreamSettings(Argus::STREAM_TYPE_EGL));
    auto* streamSettings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(settings_);
    if (!continuous_capture::check(streamSettings != nullptr, "无法获取 stream settings") ||
        !continuous_capture::check(streamSettings->supportsOutputStreamFormat(
                                       sensorMode_, Argus::PIXEL_FMT_YCbCr_420_888),
                                   "sensor mode 不支持 YUV 输出") ||
        !continuous_capture::ok(streamSettings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888),
                                "设置 YUV 格式失败") ||
        !continuous_capture::ok(streamSettings->setResolution(resolution_), "设置 YUV 分辨率失败") ||
        !continuous_capture::ok(streamSettings->setMode(Argus::EGL_STREAM_MODE_FIFO),
                                "设置 FIFO stream 失败") ||
        !continuous_capture::ok(streamSettings->setFifoLength(fifoLength_), "设置 FIFO 长度失败") ||
        !continuous_capture::ok(streamSettings->setMetadataEnable(true), "启用 metadata 失败")) return false;
    stream_.reset(iSession_->createOutputStream(settings_.get(), &status));
    if (!continuous_capture::ok(status, "创建 OutputStream 失败") ||
        !continuous_capture::check(static_cast<bool>(stream_), "OutputStream 为空")) return false;
    consumer_.reset(EGLStream::FrameConsumer::create(stream_.get()));
    iConsumer_ = Argus::interface_cast<EGLStream::IFrameConsumer>(consumer_);
    if (!continuous_capture::check(iConsumer_ != nullptr, "创建 FrameConsumer 失败")) return false;
    captureThread_ = std::thread(&ArgusCameraNode::captureLoop, this);
    return true;
}

void ArgusCameraNode::stop() {
    quit_ = true;
    if (iSession_) iSession_->stopRepeat();
    if (captureThread_.joinable()) captureThread_.join();
    if (iSession_) iSession_->waitForIdle(kFrameTimeoutNs);
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
        Argus::UniqueObj<EGLStream::Frame> frame(iConsumer_->acquireFrame(kFrameTimeoutNs, &status));
        if (!continuous_capture::ok(status, "获取 frame 失败") ||
            !continuous_capture::check(static_cast<bool>(frame), "frame 为空")) { quit_ = true; break; }
        auto* iFrame = Argus::interface_cast<EGLStream::IFrame>(frame);
        auto* image = iFrame ? iFrame->getImage() : nullptr;
        auto* image2d = image ? Argus::interface_cast<EGLStream::IImage2D>(image) : nullptr;
        if (!continuous_capture::check(image2d != nullptr, "无法读取 YUV plane stride")) { quit_ = true; break; }
        auto* argusMetadata = Argus::interface_cast<EGLStream::IArgusCaptureMetadata>(frame);
        auto* metadata = argusMetadata ? argusMetadata->getMetadata() : nullptr;
        auto* captureMetadata = metadata ? Argus::interface_cast<Argus::ICaptureMetadata>(metadata) : nullptr;
        if (!continuous_capture::check(captureMetadata != nullptr, "无法读取 Argus sensor timestamp")) {
            quit_ = true;
            break;
        }
        const uint64_t sensorTimestampNs = captureMetadata->getSensorTimestamp();
        auto packet = std::make_unique<argus_transport::ArgusFramePacket>();
        packet->header.stamp.sec = static_cast<int32_t>(sensorTimestampNs / 1000000000ULL);
        packet->header.stamp.nanosec = static_cast<uint32_t>(sensorTimestampNs % 1000000000ULL);
        packet->header.frame_id = frameId_;
        packet->frame_number = iFrame->getNumber();
        packet->width = resolution_.width();
        packet->height = resolution_.height();
        packet->y_stride = image2d->getStride(0);
        packet->u_stride = image2d->getStride(1);
        packet->v_stride = 0;
        packet->encoding = "YCbCr_420_888";
        packet->frame = std::make_shared<argus_transport::ArgusFrameOwner>(
            std::move(frame));
        publisher_->publish(std::move(packet));
        ++publishedFrames;
    }
    if (repeating) iSession_->stopRepeat();
}

}  // namespace argus_camera

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(argus_camera::ArgusCameraNode)
