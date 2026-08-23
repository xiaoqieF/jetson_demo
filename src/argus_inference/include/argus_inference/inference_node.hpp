#pragma once

#include <argus_transport/argus_frame_packet.hpp>
#include <argus_interfaces/msg/argus_inference_result.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace argus_inference {

class YoloV8Segmentation;

class InferenceNode final : public rclcpp::Node {
public:
    explicit InferenceNode(const rclcpp::NodeOptions& options);
    ~InferenceNode() override;

private:
    struct FrameSlot {
        int yuvDmabuf = -1;
        int rgbaDmabuf = -1;
        uint32_t width = 0;
        uint32_t height = 0;
        void* mappedRgbaSurface = nullptr;
        CUgraphicsResource mappedRgbaResource = nullptr;
        bool inUse = false;
    };

    struct FrameJob {
        std_msgs::msg::Header header;
        uint64_t frameNumber = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        size_t slotIndex = 0;
        float stagingMs = 0.0F;
        std::chrono::steady_clock::time_point enqueuedAt;
    };

    void stageFrame(argus_transport::ArgusFramePacket::ConstSharedPtr packet);
    void inferenceLoop();
    void inferFrame(const FrameJob& job);
    bool copyFrameToYuvBuffer(const argus_transport::ArgusFramePacket& packet, FrameSlot* slot);
    bool copyYuvToRgbaGpu(FrameSlot* slot, void** rgbaDevice, size_t* sourcePitch);
    bool ensureSlotSurfaces(FrameSlot* slot, uint32_t width, uint32_t height);
    void releaseSlotCudaInterop(FrameSlot* slot);
    void releaseSlot(FrameSlot* slot);
    bool reserveSlot(size_t* slotIndex);
    void releaseJobSlot(size_t slotIndex);

    rclcpp::Subscription<argus_transport::ArgusFramePacket>::SharedPtr subscription_;
    rclcpp::Publisher<argus_interfaces::msg::ArgusInferenceResult>::SharedPtr publisher_;
    std::unique_ptr<YoloV8Segmentation> model_;
    std::string inputTopic_;
    float confidenceThreshold_ = 0.25F;
    float iouThreshold_ = 0.45F;
    std::vector<FrameSlot> frameSlots_;
    std::deque<FrameJob> pendingJobs_;
    std::mutex jobsMutex_;
    std::condition_variable jobsReady_;
    std::thread inferenceThread_;
    bool stopInference_ = false;
    uint64_t processedFrames_ = 0;
    std::atomic<uint64_t> droppedFrames_{0};
    std::atomic<uint64_t> supersededFrames_{0};
};

}  // namespace argus_inference
