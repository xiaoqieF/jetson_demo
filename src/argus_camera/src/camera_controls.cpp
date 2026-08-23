#include "argus_camera/camera_controls.hpp"

#include <iostream>

namespace continuous_capture {

bool check(bool condition, const std::string& message) {
    if (!condition) std::cerr << "错误: " << message << '\n';
    return condition;
}

bool ok(Argus::Status status, const std::string& message) {
    if (status != Argus::STATUS_OK) {
        std::cerr << "错误: " << message << " (status=" << status << ")\n";
        return false;
    }
    return true;
}

bool configureRequest(Argus::Request* request, Argus::SensorMode* sensorMode,
                      Argus::OutputStream* stream, const Controls& controls) {
    Argus::IRequest* iRequest = Argus::interface_cast<Argus::IRequest>(request);
    Argus::ISourceSettings* iSource = Argus::interface_cast<Argus::ISourceSettings>(request);
    Argus::IAutoControlSettings* iAuto = nullptr;
    Argus::IDenoiseSettings* iDenoise = Argus::interface_cast<Argus::IDenoiseSettings>(request);
    Argus::IEdgeEnhanceSettings* iEdge = Argus::interface_cast<Argus::IEdgeEnhanceSettings>(request);
    if (iRequest) {
        iAuto = Argus::interface_cast<Argus::IAutoControlSettings>(iRequest->getAutoControlSettings());
    }
    if (!check(iRequest && iSource && iAuto && iDenoise && iEdge, "request 控制接口不可用")) return false;
    if (!ok(iRequest->enableOutputStream(stream), "启用 YUV 输出失败") ||
        !ok(iSource->setSensorMode(sensorMode), "设置 sensor mode 失败") ||
        !ok(iAuto->setColorSaturationEnable(true), "启用饱和度控制失败") ||
        !ok(iAuto->setColorSaturation(controls.saturation), "设置饱和度失败") ||
        !ok(iAuto->setExposureCompensation(controls.exposureCompensation), "设置曝光补偿失败") ||
        !ok(iAuto->setIspDigitalGainRange(Argus::Range<float>(controls.ispDigitalGain)), "设置 ISP digital gain 失败") ||
        !ok(iDenoise->setDenoiseMode(controls.denoiseMode), "设置降噪模式失败") ||
        !ok(iDenoise->setDenoiseStrength(controls.denoiseStrength), "设置降噪强度失败") ||
        !ok(iEdge->setEdgeEnhanceMode(controls.edgeMode), "设置边缘增强模式失败") ||
        !ok(iEdge->setEdgeEnhanceStrength(controls.edgeStrength), "设置边缘增强强度失败")) return false;
    if (controls.manualWb) {
        if (!ok(iAuto->setWbGains(controls.wbGains), "设置手动白平衡增益失败") ||
            !ok(iAuto->setAwbMode(Argus::AWB_MODE_MANUAL), "设置手动白平衡模式失败")) return false;
    } else if (!ok(iAuto->setAwbMode(Argus::AWB_MODE_AUTO), "恢复自动白平衡失败")) return false;
    return true;
}

}  // namespace continuous_capture
