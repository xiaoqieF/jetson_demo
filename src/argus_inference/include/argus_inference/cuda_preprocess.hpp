#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace argus_inference {

bool preprocessRgbaOnGpu(const void* rgbaDevice, size_t sourcePitch, int sourceWidth,
                         int sourceHeight, float* tensorDevice, int inputSize,
                         cudaStream_t stream);

}  // namespace argus_inference
