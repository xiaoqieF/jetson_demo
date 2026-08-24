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
    struct StagingSlot {
        enum class State {
            kFree,
            kFilling,
            kQueued,
            kProcessing,
        };

        int yuvDmabuf = -1;
        int rgbaDmabuf = -1;
        uint32_t width = 0;
        uint32_t height = 0;
        void* mappedRgbaSurface = nullptr;
        CUgraphicsResource mappedRgbaResource = nullptr;
        std_msgs::msg::Header header;
        uint64_t frameNumber = 0;
        float stagingMs = 0.0F;
        std::chrono::steady_clock::time_point enqueuedAt;
        State state = State::kFree;
    };

    void stageFrame(argus_transport::ArgusFramePacket::ConstSharedPtr packet);
    void inferenceLoop();
    void inferFrame(size_t slotIndex);
    bool copyFrameToYuvBuffer(const argus_transport::ArgusFramePacket& packet, StagingSlot* slot);
    bool copyYuvToRgbaGpu(StagingSlot* slot, void** rgbaDevice, size_t* sourcePitch);
    bool ensureSlotSurfaces(StagingSlot* slot, uint32_t width, uint32_t height);
    void releaseSlotCudaInterop(StagingSlot* slot);
    void releaseSlot(StagingSlot* slot);
    bool reserveSlot(size_t* slotIndex);
    void releaseSlotForReuse(size_t slotIndex);

    rclcpp::Subscription<argus_transport::ArgusFramePacket>::SharedPtr subscription_;
    rclcpp::Publisher<argus_interfaces::msg::ArgusInferenceResult>::SharedPtr publisher_;
    std::unique_ptr<YoloV8Segmentation> model_;
    std::string inputTopic_;
    int inputSize_ = 640;
    float confidenceThreshold_ = 0.25F;
    float iouThreshold_ = 0.45F;
    std::vector<StagingSlot> frameSlots_;
    std::deque<size_t> readySlots_;
    std::mutex jobsMutex_;
    std::condition_variable jobsReady_;
    std::thread inferenceThread_;
    bool stopInference_ = false;
    uint64_t processedFrames_ = 0;
    std::atomic<uint64_t> droppedFrames_{0};
    std::atomic<uint64_t> supersededFrames_{0};
};

}  // namespace argus_inference
