#include "argus_inference/cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

namespace argus_inference {
namespace {

__global__ void rgbaToRgbPlanar(const uchar4* source, size_t sourcePitch,
                                float* output, int inputSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inputSize || y >= inputSize) return;
    const auto* row = reinterpret_cast<const uchar4*>(
        reinterpret_cast<const unsigned char*>(source) + y * sourcePitch);
    const uchar4 value = row[x];
    const int pixel = y * inputSize + x;
    output[pixel] = value.x / 255.0F;
    output[inputSize * inputSize + pixel] = value.y / 255.0F;
    output[2 * inputSize * inputSize + pixel] = value.z / 255.0F;
}

__device__ float bilinearSample(const float* source, int width, int height, float x, float y) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
    const int left = static_cast<int>(floorf(x));
    const int top = static_cast<int>(floorf(y));
    const int right = min(left + 1, width - 1);
    const int bottom = min(top + 1, height - 1);
    const float horizontal = x - left;
    const float vertical = y - top;
    const float topValue = source[top * width + left] * (1.0F - horizontal) +
                           source[top * width + right] * horizontal;
    const float bottomValue = source[bottom * width + left] * (1.0F - horizontal) +
                              source[bottom * width + right] * horizontal;
    return topValue * (1.0F - vertical) + bottomValue * vertical;
}

__global__ void decodeMasks(const float* prototype, const float* coefficients,
                            const MaskDecodeRoi* rois, int maskDimensions,
                            int prototypeHeight, int prototypeWidth, int sourceWidth,
                            int sourceHeight, int inputSize, int paddingX, int paddingY,
                            unsigned char* masks) {
    const int roiIndex = blockIdx.z;
    const MaskDecodeRoi roi = rois[roiIndex];
    const int localX = blockIdx.x * blockDim.x + threadIdx.x;
    const int localY = blockIdx.y * blockDim.y + threadIdx.y;
    if (localX >= roi.width || localY >= roi.height) return;

    const int sourceX = roi.x + localX;
    const int sourceY = roi.y + localY;
    const int validWidth = inputSize - 2 * paddingX;
    const int validHeight = inputSize - 2 * paddingY;
    const float prototypeX =
        ((static_cast<float>(sourceX) + 0.5F) * validWidth / sourceWidth + paddingX) *
            prototypeWidth / inputSize -
        0.5F;
    const float prototypeY =
        ((static_cast<float>(sourceY) + 0.5F) * validHeight / sourceHeight + paddingY) *
            prototypeHeight / inputSize -
        0.5F;
    const float* roiCoefficients = coefficients + static_cast<size_t>(roiIndex) * maskDimensions;
    const size_t prototypePlane = static_cast<size_t>(prototypeHeight) * prototypeWidth;
    float value = 0.0F;
    for (int channel = 0; channel < maskDimensions; ++channel) {
        value += roiCoefficients[channel] *
                 bilinearSample(prototype + static_cast<size_t>(channel) * prototypePlane,
                                prototypeWidth, prototypeHeight, prototypeX, prototypeY);
    }
    const float probability = 1.0F / (1.0F + expf(-value));
    masks[roi.outputOffset + static_cast<size_t>(localY) * roi.width + localX] =
        probability > 0.5F ? 255 : 0;
}

}  // namespace

bool preprocessRgbaOnGpu(const void* rgbaDevice, size_t sourcePitch, int sourceWidth,
                         int sourceHeight, float* tensorDevice, int inputSize,
                         cudaStream_t stream) {
    if (!rgbaDevice || !tensorDevice || sourcePitch < static_cast<size_t>(inputSize) * sizeof(uchar4) ||
        sourceWidth != inputSize || sourceHeight != inputSize || inputSize <= 0) return false;
    constexpr dim3 threads(16, 16);
    const dim3 blocks((inputSize + threads.x - 1) / threads.x,
                      (inputSize + threads.y - 1) / threads.y);
    rgbaToRgbPlanar<<<blocks, threads, 0, stream>>>(
        static_cast<const uchar4*>(rgbaDevice), sourcePitch, tensorDevice, inputSize);
    return cudaGetLastError() == cudaSuccess;
}

GpuMaskDecoder::~GpuMaskDecoder() {
    releaseBuffers();
}

bool GpuMaskDecoder::ensureBuffer(void** buffer, size_t* capacity, size_t bytes) {
    if (!buffer || !capacity) return false;
    if (bytes <= *capacity) return true;
    if (*buffer && cudaFree(*buffer) != cudaSuccess) return false;
    *buffer = nullptr;
    *capacity = 0;
    if (cudaMalloc(buffer, bytes) != cudaSuccess) return false;
    *capacity = bytes;
    return true;
}

void GpuMaskDecoder::releaseBuffers() {
    if (deviceCoefficients_) cudaFree(deviceCoefficients_);
    deviceCoefficients_ = nullptr;
    coefficientCapacity_ = 0;
    if (deviceRois_) cudaFree(deviceRois_);
    deviceRois_ = nullptr;
    roiCapacity_ = 0;
    if (deviceMasks_) cudaFree(deviceMasks_);
    deviceMasks_ = nullptr;
    maskCapacity_ = 0;
}

bool GpuMaskDecoder::decode(const float* prototypeDevice, int maskDimensions, int prototypeHeight,
                            int prototypeWidth, const std::vector<float>& coefficients,
                            const std::vector<MaskDecodeRoi>& rois, int sourceWidth,
                            int sourceHeight, int inputSize, int paddingX, int paddingY,
                            cudaStream_t stream, std::vector<unsigned char>* masks) {
    if (!prototypeDevice || !masks || !stream || maskDimensions <= 0 || prototypeHeight <= 0 ||
        prototypeWidth <= 0 || sourceWidth <= 0 || sourceHeight <= 0 || inputSize <= 0 ||
        paddingX < 0 || paddingY < 0 || inputSize <= 2 * paddingX || inputSize <= 2 * paddingY ||
        rois.empty() || rois.size() > std::numeric_limits<unsigned int>::max() ||
        coefficients.size() != rois.size() * static_cast<size_t>(maskDimensions)) return false;

    size_t maskBytes = 0;
    int maximumWidth = 0;
    int maximumHeight = 0;
    for (const auto& roi : rois) {
        if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
            roi.x + roi.width > sourceWidth || roi.y + roi.height > sourceHeight ||
            roi.outputOffset != maskBytes ||
            static_cast<size_t>(roi.width) > std::numeric_limits<size_t>::max() / roi.height) {
            return false;
        }
        const size_t roiBytes = static_cast<size_t>(roi.width) * roi.height;
        if (maskBytes > std::numeric_limits<size_t>::max() - roiBytes) return false;
        maskBytes += roiBytes;
        maximumWidth = std::max(maximumWidth, roi.width);
        maximumHeight = std::max(maximumHeight, roi.height);
    }
    if (!ensureBuffer(&deviceCoefficients_, &coefficientCapacity_, coefficients.size() * sizeof(float)) ||
        !ensureBuffer(&deviceRois_, &roiCapacity_, rois.size() * sizeof(MaskDecodeRoi)) ||
        !ensureBuffer(&deviceMasks_, &maskCapacity_, maskBytes)) return false;

    masks->resize(maskBytes);
    if (cudaMemcpyAsync(deviceCoefficients_, coefficients.data(), coefficients.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(deviceRois_, rois.data(), rois.size() * sizeof(MaskDecodeRoi),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
    constexpr dim3 threads(16, 16);
    const dim3 blocks((maximumWidth + threads.x - 1) / threads.x,
                      (maximumHeight + threads.y - 1) / threads.y,
                      static_cast<unsigned int>(rois.size()));
    decodeMasks<<<blocks, threads, 0, stream>>>(
        prototypeDevice, static_cast<const float*>(deviceCoefficients_),
        static_cast<const MaskDecodeRoi*>(deviceRois_), maskDimensions, prototypeHeight,
        prototypeWidth, sourceWidth, sourceHeight, inputSize, paddingX, paddingY,
        static_cast<unsigned char*>(deviceMasks_));
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(masks->data(), deviceMasks_, maskBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;
    return true;
}

}  // namespace argus_inference
