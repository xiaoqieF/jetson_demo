#pragma once

#include <Argus/Argus.h>

#include <string>

namespace continuous_capture {

constexpr uint64_t kTimeoutNs = 5ULL * 1000ULL * 1000ULL * 1000ULL;

struct Controls {
    float saturation = 1.0f;
    float exposureCompensation = 0.0f;
    float ispDigitalGain = 1.0f;
    float denoiseStrength = 1.0f;
    float edgeStrength = 1.0f;
    Argus::DenoiseMode denoiseMode = Argus::DENOISE_MODE_FAST;
    Argus::EdgeEnhanceMode edgeMode = Argus::EDGE_ENHANCE_MODE_FAST;
    bool manualWb = false;
    Argus::BayerTuple<float> wbGains = Argus::BayerTuple<float>(1.0f);
    double frameRate = 0.0;
};

bool check(bool condition, const std::string& message);
bool ok(Argus::Status status, const std::string& message);
bool configureRequest(Argus::Request* request, Argus::SensorMode* sensorMode,
                      Argus::OutputStream* stream, const Controls& controls);

}  // namespace continuous_capture
