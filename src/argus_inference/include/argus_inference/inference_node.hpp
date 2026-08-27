#pragma once

#include <argus_transport/argus_frame_packet.hpp>
#include <argus_inference/yolov8_segmentation.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace argus_inference {

class InferenceNode final : public rclcpp::Node {
public:
    explicit InferenceNode(const rclcpp::NodeOptions& options);
    ~InferenceNode() override;

private:
    struct StagingSlot {
        int rgbaDmabuf = -1;
        uint32_t width = 0;
        uint32_t height = 0;
        std::shared_ptr<argus_transport::ArgusFrameOwner> sourceFrame;
        void* mappedRgbaSurface = nullptr;
        CUgraphicsResource mappedRgbaResource = nullptr;
        void* rgbaDevice = nullptr;
        size_t rgbaPitch = 0;
        std_msgs::msg::Header header;
        uint64_t frameNumber = 0;
    };

    struct PendingFrame {
        std_msgs::msg::Header header;
        uint64_t frameNumber = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        std::shared_ptr<argus_transport::ArgusFrameOwner> sourceFrame;
    };

    void stageFrame(argus_transport::ArgusFramePacket::ConstSharedPtr packet);
    void inferenceLoop();
    void inferFrame(PendingFrame frame);
    // 发布示例分割可视化，十分耗时
    bool publishOverlay(const StagingSlot& slot,
                       const std::vector<SegmentationInstance>& instances);
    bool copyYuvToRgbaGpu(StagingSlot* slot, void** rgbaDevice, size_t* sourcePitch);
    bool initializeSlotSurface(StagingSlot* slot);
    bool initializeSlotCudaInterop(StagingSlot* slot);
    void releaseSlotCudaInterop(StagingSlot* slot);
    void releaseSlot(StagingSlot* slot);

    rclcpp::Subscription<argus_transport::ArgusFramePacket>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
    std::unique_ptr<YoloV8Segmentation> model_;
    std::string inputTopic_;
    int inputSize_ = 640;
    uint64_t timingLogEveryNFrames_ = 30;
    float confidenceThreshold_ = 0.25F;
    float iouThreshold_ = 0.45F;
    int overlayQuality_ = 90;
    bool enableOverlay_ = false;
    StagingSlot inferenceSlot_;
    std::optional<PendingFrame> pendingFrame_;
    std::mutex jobsMutex_;
    std::condition_variable jobsReady_;
    std::thread inferenceThread_;
    bool stopInference_ = false;
    uint64_t processedFrames_ = 0;
    std::atomic<uint64_t> supersededFrames_{0};
};

}  // namespace argus_inference
