#include "argus_inference/cuda_preprocess.hpp"

#include <cuda_runtime.h>

namespace argus_inference {
namespace {

__global__ void rgbaToRgbPlanar(const uchar4* source, size_t sourcePitch,
                                 float* output, int inputSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inputSize || y >= inputSize) return;
    const auto* row = reinterpret_cast<const uchar4*>(reinterpret_cast<const unsigned char*>(source) + y * sourcePitch);
    const uchar4 value = row[x];
    const int pixel = y * inputSize + x;
    output[pixel] = value.x / 255.0F;
    output[inputSize * inputSize + pixel] = value.y / 255.0F;
    output[2 * inputSize * inputSize + pixel] = value.z / 255.0F;
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

}  // namespace argus_inference
