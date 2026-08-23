#pragma once

#include <argus_interfaces/msg/argus_yuv_frame.hpp>
#include <argus_transport/argus_frame_packet.hpp>
#include <rclcpp/rclcpp.hpp>

#include "argus_camera/camera_controls.hpp"

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/FrameConsumer.h>

#include <atomic>
#include <memory>
#include <thread>

namespace argus_camera {

class ArgusCameraNode final : public rclcpp::Node {
public:
    explicit ArgusCameraNode(const rclcpp::NodeOptions& options);
    ~ArgusCameraNode() override;

private:
    bool start();
    void stop();
    void captureLoop();

    rclcpp::Publisher<argus_transport::ArgusFramePacket>::SharedPtr publisher_;
    int64_t requestedFrames_ = 0;
    int cameraIndex_ = 0;
    int sensorModeIndex_ = 0;
    int fifoLength_ = 8;
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
    Argus::UniqueObj<EGLStream::FrameConsumer> consumer_;
    EGLStream::IFrameConsumer* iConsumer_ = nullptr;
};

}  // namespace argus_camera
