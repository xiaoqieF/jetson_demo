#pragma once

#include <argus_transport/argus_frame_packet.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <NvJpegEncoder.h>

#include <memory>

namespace argus_visualization {

class VisualizationNode final : public rclcpp::Node {
public:
    explicit VisualizationNode(const rclcpp::NodeOptions& options);
    ~VisualizationNode() override;

private:
    void publishJpeg(const argus_transport::ArgusFramePacket& packet);

    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
    rclcpp::Subscription<argus_transport::ArgusFramePacket>::SharedPtr subscription_;
    std::unique_ptr<NvJPEGEncoder> encoder_;
    int dmabuf_ = -1;
};

}  // namespace argus_visualization
