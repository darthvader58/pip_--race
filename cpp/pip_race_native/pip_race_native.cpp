#include <algorithm>
#include <cmath>
#include <cstddef>

#if defined(_WIN32)
#define PIP_RACE_EXPORT extern "C" __declspec(dllexport)
#else
#define PIP_RACE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr std::size_t kMinFeatureDim = 16;
constexpr std::size_t kOutputDim = 3;

inline float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

PIP_RACE_EXPORT int pip_race_predict_v1(
    const float* features,
    std::size_t batch_size,
    std::size_t feature_dim,
    float* outputs
) {
    if (features == nullptr || outputs == nullptr || feature_dim < kMinFeatureDim) {
        return -1;
    }

    for (std::size_t row_idx = 0; row_idx < batch_size; ++row_idx) {
        const float* row = features + row_idx * feature_dim;
        const float degradation = std::min(1.0f, std::max(0.0f, row[14]));
        const float score = row[3] * 0.08f + row[11] * 0.65f + row[14] * 0.75f;
        const float pit_risk = sigmoid(score);
        const float confidence = std::max(pit_risk, 1.0f - pit_risk);

        float* out = outputs + row_idx * kOutputDim;
        out[0] = pit_risk;
        out[1] = degradation;
        out[2] = confidence;
    }

    return 0;
}

PIP_RACE_EXPORT std::size_t pip_race_output_dim_v1() {
    return kOutputDim;
}
