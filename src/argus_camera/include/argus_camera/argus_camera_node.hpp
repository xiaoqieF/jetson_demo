#pragma once

#include <argus_interfaces/msg/argus_yuv_frame.hpp>
#include <argus_transport/argus_frame_packet.hpp>
#include <rclcpp/rclcpp.hpp>

#include "argus_camera/camera_controls.hpp"

#include <Argus/Argus.h>

#include <EGL/egl.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

struct NvBufSurface;

namespace argus_camera {

class ArgusCameraNode final : public rclcpp::Node {
public:
    explicit ArgusCameraNode(const rclcpp::NodeOptions& options);
    ~ArgusCameraNode() override;

private:
    struct CaptureBufferInfo {
        int dmabuf = -1;
        uint32_t yStride = 0;
        uint32_t uStride = 0;
        uint32_t vStride = 0;
    };

    bool start();
    void stop();
    void captureLoop();
    bool createCaptureBufferPool();
    void destroyCaptureBufferPool();

    rclcpp::Publisher<argus_transport::ArgusFramePacket>::SharedPtr publisher_;
    int64_t requestedFrames_ = 0;
    int cameraIndex_ = 0;
    int sensorModeIndex_ = 0;
    int captureBufferCount_ = 4;
    double frameRate_ = 0.0;
    std::string frameId_;
    std::atomic<bool> quit_{false};
    std::atomic<bool> started_{false};
    continuous_capture::Controls controls_;
    std::thread captureThread_;
    Argus::UniqueObj<Argus::CameraProvider> provider_;
    Argus::ICameraProvider* iProvider_ = nullptr;
    Argus::UniqueObj<Argus::CaptureSession> session_;
    Argus::ICaptureSession* iSession_ = nullptr;
    Argus::SensorMode* sensorMode_ = nullptr;
    Argus::Size2D<uint32_t> resolution_;
    Argus::UniqueObj<Argus::OutputStreamSettings> settings_;
    Argus::UniqueObj<Argus::OutputStream> stream_;
    Argus::IBufferOutputStream* iBufferStream_ = nullptr;
    Argus::UniqueObj<Argus::BufferSettings> bufferSettings_;
    std::vector<Argus::UniqueObj<Argus::Buffer>> captureBuffers_;
    std::vector<CaptureBufferInfo> captureBufferInfos_;
    std::vector<NvBufSurface*> captureSurfaces_;
    EGLDisplay eglDisplay_ = EGL_NO_DISPLAY;
    bool eglInitialized_ = false;
    std::shared_ptr<argus_transport::ArgusBufferReleaseState> bufferReleaseState_;
};

}  // namespace argus_camera
