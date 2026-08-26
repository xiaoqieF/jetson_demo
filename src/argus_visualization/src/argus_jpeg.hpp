#pragma once

#include <NvJpegEncoder.h>

#include <cstdint>
#include <vector>

namespace argus_pipeline {

bool encodeFrameToJpeg(int dmabuf, NvJPEGEncoder* encoder,
                       uint32_t width, uint32_t height,
                       std::vector<uint8_t>* jpeg);

}  // namespace argus_pipeline
