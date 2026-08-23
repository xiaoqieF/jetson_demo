#include "argus_inference/cuda_preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace argus_inference {
namespace {

__device__ uchar4 bilinearSample(const uchar4* image, size_t pitch, int width, int height,
                                 float x, float y) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float dx = x - x0;
    const float dy = y - y0;
    const auto* row0 = reinterpret_cast<const uchar4*>(reinterpret_cast<const unsigned char*>(image) + y0 * pitch);
    const auto* row1 = reinterpret_cast<const uchar4*>(reinterpret_cast<const unsigned char*>(image) + y1 * pitch);
    const uchar4 p00 = row0[x0];
    const uchar4 p01 = row0[x1];
    const uchar4 p10 = row1[x0];
    const uchar4 p11 = row1[x1];
    uchar4 result;
    result.x = static_cast<unsigned char>((1.0F - dy) * ((1.0F - dx) * p00.x + dx * p01.x) +
                                           dy * ((1.0F - dx) * p10.x + dx * p11.x));
    result.y = static_cast<unsigned char>((1.0F - dy) * ((1.0F - dx) * p00.y + dx * p01.y) +
                                           dy * ((1.0F - dx) * p10.y + dx * p11.y));
    result.z = static_cast<unsigned char>((1.0F - dy) * ((1.0F - dx) * p00.z + dx * p01.z) +
                                           dy * ((1.0F - dx) * p10.z + dx * p11.z));
    result.w = 255;
    return result;
}

__global__ void letterboxRgbaToRgbPlanar(const uchar4* source, size_t sourcePitch,
                                         int sourceWidth, int sourceHeight, float* output,
                                         int inputSize, float scale, int paddingX, int paddingY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inputSize || y >= inputSize) return;
    const int pixel = y * inputSize + x;
    float red = 114.0F / 255.0F;
    float green = red;
    float blue = red;
    if (x >= paddingX && y >= paddingY && x < paddingX + static_cast<int>(sourceWidth * scale) &&
        y < paddingY + static_cast<int>(sourceHeight * scale)) {
        const float sourceX = (static_cast<float>(x - paddingX) + 0.5F) / scale - 0.5F;
        const float sourceY = (static_cast<float>(y - paddingY) + 0.5F) / scale - 0.5F;
        const uchar4 value = bilinearSample(source, sourcePitch, sourceWidth, sourceHeight, sourceX, sourceY);
        red = value.x / 255.0F;
        green = value.y / 255.0F;
        blue = value.z / 255.0F;
    }
    output[pixel] = red;
    output[inputSize * inputSize + pixel] = green;
    output[2 * inputSize * inputSize + pixel] = blue;
}

}  // namespace

bool preprocessRgbaOnGpu(const void* rgbaDevice, size_t sourcePitch, int sourceWidth,
                         int sourceHeight, float* tensorDevice, int inputSize,
                         cudaStream_t stream) {
    if (!rgbaDevice || !tensorDevice || sourcePitch < static_cast<size_t>(sourceWidth) * sizeof(uchar4) ||
        sourceWidth <= 0 || sourceHeight <= 0 || inputSize <= 0) return false;
    const float scale = fminf(static_cast<float>(inputSize) / sourceWidth,
                              static_cast<float>(inputSize) / sourceHeight);
    const int scaledWidth = static_cast<int>(roundf(sourceWidth * scale));
    const int scaledHeight = static_cast<int>(roundf(sourceHeight * scale));
    const int paddingX = (inputSize - scaledWidth) / 2;
    const int paddingY = (inputSize - scaledHeight) / 2;
    constexpr dim3 threads(16, 16);
    const dim3 blocks((inputSize + threads.x - 1) / threads.x,
                      (inputSize + threads.y - 1) / threads.y);
    letterboxRgbaToRgbPlanar<<<blocks, threads, 0, stream>>>(
        static_cast<const uchar4*>(rgbaDevice), sourcePitch, sourceWidth, sourceHeight,
        tensorDevice, inputSize, scale, paddingX, paddingY);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace argus_inference
