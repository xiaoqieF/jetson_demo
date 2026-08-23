#include "argus_jpeg.hpp"

#include <EGLStream/NV/ImageNativeBuffer.h>
#include <EGLStream/Frame.h>
#include <EGLStream/Image.h>
#include <NvBufSurface.h>

#include <iostream>
#include <memory>

namespace argus_pipeline {
namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "错误: " << message << '\n';
    return condition;
}

bool ok(Argus::Status status, const char* message) {
    if (status != Argus::STATUS_OK) {
        std::cerr << "错误: " << message << " (status=" << status << ")\n";
        return false;
    }
    return true;
}

}  // namespace

bool encodeFrameToJpeg(EGLStream::Frame* frame, NvJPEGEncoder* encoder,
                       int* dmabuf, uint32_t width, uint32_t height,
                       std::vector<uint8_t>* jpeg) {
    if (!frame || !encoder || !dmabuf || !jpeg) {
        return check(false, "JPEG 编码参数无效");
    }
    auto* iFrame = Argus::interface_cast<EGLStream::IFrame>(
        static_cast<Argus::InterfaceProvider*>(frame));
    auto* image = iFrame ? iFrame->getImage() : nullptr;
    auto* nativeBuffer = image
        ? Argus::interface_cast<EGLStream::NV::IImageNativeBuffer>(image)
        : nullptr;
    if (!check(nativeBuffer != nullptr, "Argus image 不支持 native buffer")) {
        return false;
    }
    if (*dmabuf < 0) {
        *dmabuf = nativeBuffer->createNvBuffer(
            Argus::Size2D<uint32_t>(width, height), NVBUF_COLOR_FORMAT_YUV420,
            NVBUF_LAYOUT_PITCH);
        if (!check(*dmabuf >= 0, "创建 JPEG dmabuf 失败")) {
            return false;
        }
    } else if (!ok(nativeBuffer->copyToNvBuffer(*dmabuf),
                   "复制 Argus image 到 JPEG dmabuf 失败")) {
        return false;
    }
    const unsigned long outputCapacity =
        static_cast<unsigned long>(width) * height * 3 / 2;
    auto output = std::make_unique<unsigned char[]>(outputCapacity);
    auto* outputData = output.get();
    unsigned long outputSize = outputCapacity;
    if (encoder->encodeFromFd(*dmabuf, JCS_YCbCr, &outputData, outputSize, 85) != 0) {
        return check(false, "JPEG 编码失败");
    }
    if (!check(outputData == output.get(), "JPEG 输出超出预分配缓冲区")) return false;
    jpeg->assign(outputData, outputData + outputSize);
    return true;
}

}  // namespace argus_pipeline
