#pragma once

#include <argus_interfaces/msg/argus_yuv_frame.hpp>

#include <Argus/Argus.h>

#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <utility>

#include <rclcpp/type_adapter.hpp>
#include <std_msgs/msg/header.hpp>

namespace argus_transport {

class ArgusBufferReleaseState final {
public:
    void setStream(Argus::IBufferOutputStream* stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_ = stream;
    }

    void disable() {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_ = nullptr;
    }

    void release(Argus::Buffer* buffer) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stream_ && buffer) stream_->releaseBuffer(buffer);
    }

private:
    std::mutex mutex_;
    Argus::IBufferOutputStream* stream_ = nullptr;
};

class ArgusFrameOwner {
public:
    ArgusFrameOwner(int dmabuf, Argus::Buffer* buffer,
                    std::shared_ptr<ArgusBufferReleaseState> releaseState)
        : dmabuf_(dmabuf), buffer_(buffer), releaseState_(std::move(releaseState)) {}

    ~ArgusFrameOwner() {
        if (releaseState_) releaseState_->release(buffer_);
    }

    int dmabuf() const { return dmabuf_; }

private:
    int dmabuf_ = -1;
    Argus::Buffer* buffer_ = nullptr;
    std::shared_ptr<ArgusBufferReleaseState> releaseState_;
};

struct ArgusFramePacket {
    using SharedPtr = std::shared_ptr<ArgusFramePacket>;
    using ConstSharedPtr = std::shared_ptr<const ArgusFramePacket>;

    std_msgs::msg::Header header;
    uint64_t frame_number = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t y_stride = 0;
    uint32_t u_stride = 0;
    uint32_t v_stride = 0;
    std::string encoding;
    std::shared_ptr<ArgusFrameOwner> frame;
};

}  // namespace argus_transport

namespace rclcpp {

template<>
struct TypeAdapter<
    argus_transport::ArgusFramePacket,
    argus_interfaces::msg::ArgusYuvFrame> {
    using is_specialized = std::true_type;
    using custom_type = argus_transport::ArgusFramePacket;
    using ros_message_type = argus_interfaces::msg::ArgusYuvFrame;

    static void convert_to_ros_message(
        const custom_type& source, ros_message_type& target) {
        target.header = source.header;
        target.frame_number = source.frame_number;
        target.width = source.width;
        target.height = source.height;
        target.y_stride = source.y_stride;
        target.u_stride = source.u_stride;
        target.v_stride = source.v_stride;
        target.encoding = source.encoding;
    }

    static void convert_to_custom(
        const ros_message_type& source, custom_type& target) {
        target.header = source.header;
        target.frame_number = source.frame_number;
        target.width = source.width;
        target.height = source.height;
        target.y_stride = source.y_stride;
        target.u_stride = source.u_stride;
        target.v_stride = source.v_stride;
        target.encoding = source.encoding;
        target.frame.reset();
    }
};

}  // namespace rclcpp

RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(
    argus_transport::ArgusFramePacket,
    argus_interfaces::msg::ArgusYuvFrame);
