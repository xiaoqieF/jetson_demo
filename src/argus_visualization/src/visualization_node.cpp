#include "argus_visualization/visualization_node.hpp"

#include "argus_jpeg.hpp"

#include <NvBufSurface.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace argus_visualization {

VisualizationNode::VisualizationNode(const rclcpp::NodeOptions& options)
    : Node("yuv_visualization_node", options) {
    const auto inputTopic = declare_parameter<std::string>("input_topic", "/camera/image/yuv");
    const auto outputTopic = declare_parameter<std::string>(
        "output_topic", "/camera/image/compressed");
    publisher_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        outputTopic, rclcpp::SensorDataQoS());
    encoder_.reset(NvJPEGEncoder::createJPEGEncoder("argus_visualization_jpeg"));
    if (!encoder_) throw std::runtime_error("创建 NvJPEGEncoder 失败");
    subscription_ = create_subscription<argus_transport::ArgusFramePacket>(
        inputTopic, rclcpp::QoS(rclcpp::KeepLast(8)).reliable(),
        [this](argus_transport::ArgusFramePacket::ConstSharedPtr message) {
            publishJpeg(*message);
        });
}

VisualizationNode::~VisualizationNode() {
    if (dmabuf_ >= 0) NvBufSurf::NvDestroy(dmabuf_);
}

void VisualizationNode::publishJpeg(
    const argus_transport::ArgusFramePacket& packet) {
    if (!packet.frame) {
        RCLCPP_WARN(get_logger(), "收到不含 native frame 的 Argus YUV packet");
        return;
    }
    std::vector<uint8_t> jpeg;
    if (!argus_pipeline::encodeFrameToJpeg(
            packet.frame->get(), encoder_.get(), &dmabuf_, packet.width, packet.height, &jpeg)) return;
    sensor_msgs::msg::CompressedImage compressed;
    compressed.header = packet.header;
    compressed.format = "jpeg";
    compressed.data = std::move(jpeg);
    publisher_->publish(compressed);
}

}  // namespace argus_visualization

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(argus_visualization::VisualizationNode)
