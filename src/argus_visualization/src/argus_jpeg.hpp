#pragma once

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <NvJpegEncoder.h>

#include <cstdint>
#include <vector>

namespace argus_pipeline {

bool encodeFrameToJpeg(EGLStream::Frame* frame, NvJPEGEncoder* encoder,
                       int* dmabuf, uint32_t width, uint32_t height,
                       std::vector<uint8_t>* jpeg);

}  // namespace argus_pipeline
