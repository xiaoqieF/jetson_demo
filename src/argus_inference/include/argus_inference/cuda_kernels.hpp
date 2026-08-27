#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

namespace argus_inference {

bool preprocessRgbaOnGpu(const void* rgbaDevice, size_t sourcePitch, int sourceWidth,
                         int sourceHeight, float* tensorDevice, int inputSize,
                         cudaStream_t stream);

struct MaskDecodeRoi {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    size_t outputOffset = 0;
};

class GpuMaskDecoder final {
public:
    GpuMaskDecoder() = default;
    ~GpuMaskDecoder();

    GpuMaskDecoder(const GpuMaskDecoder&) = delete;
    GpuMaskDecoder& operator=(const GpuMaskDecoder&) = delete;

    bool decode(const float* prototypeDevice, int maskDimensions, int prototypeHeight,
                int prototypeWidth, const std::vector<float>& coefficients,
                const std::vector<MaskDecodeRoi>& rois, int sourceWidth, int sourceHeight,
                int inputSize, int paddingX, int paddingY, cudaStream_t stream,
                std::vector<unsigned char>* masks);

private:
    bool ensureBuffer(void** buffer, size_t* capacity, size_t bytes);
    void releaseBuffers();

    void* deviceCoefficients_ = nullptr;
    size_t coefficientCapacity_ = 0;
    void* deviceRois_ = nullptr;
    size_t roiCapacity_ = 0;
    void* deviceMasks_ = nullptr;
    size_t maskCapacity_ = 0;
};

}  // namespace argus_inference
