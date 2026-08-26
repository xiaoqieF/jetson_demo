#include "argus_jpeg.hpp"

#include <iostream>
#include <memory>

namespace argus_pipeline {
namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "错误: " << message << '\n';
    return condition;
}

}  // namespace

bool encodeFrameToJpeg(int dmabuf, NvJPEGEncoder* encoder,
                       uint32_t width, uint32_t height,
                       std::vector<uint8_t>* jpeg) {
    if (dmabuf < 0 || !encoder || !jpeg) {
        return check(false, "JPEG 编码参数无效");
    }
    const unsigned long outputCapacity =
        static_cast<unsigned long>(width) * height * 3 / 2;
    auto output = std::make_unique<unsigned char[]>(outputCapacity);
    auto* outputData = output.get();
    unsigned long outputSize = outputCapacity;
    if (encoder->encodeFromFd(dmabuf, JCS_YCbCr, &outputData, outputSize, 85) != 0) {
        return check(false, "JPEG 编码失败");
    }
    if (!check(outputData == output.get(), "JPEG 输出超出预分配缓冲区")) return false;
    jpeg->assign(outputData, outputData + outputSize);
    return true;
}

}  // namespace argus_pipeline
