#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

/**
 * @brief Root Mean Square Normalization (RMS Norm)
 *       Yi = Wi * Xi / sqrt(Sum(Xj^2)/n + eps
 * 
 * @tparam T 
 * @param out [batch_size, out_width]
 * @param in [batch_size, in_width]
 * @param weight [in_width]
 * @param eps float
 * @param batch_size 
 * @param in_width 
 * @param out_width 
 */
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, const float eps, size_t batch_size, size_t in_width, size_t out_width) {
    for (size_t b = 0; b < batch_size; ++b) {
        const T *in_row = in + b * in_width;
        T *out_row = out + b * out_width;
        float sum_sq = 0.0f;
        for (size_t i = 0; i < in_width; ++i) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val_f = llaisys::utils::cast<float>(in_row[i]);
                sum_sq += val_f * val_f;
            } else {
                sum_sq += in_row[i] * in_row[i];
            }
        }
        float rms;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            rms = std::sqrt(sum_sq / llaisys::utils::cast<float>(in_width) + eps);
        } else {
            rms = std::sqrt(sum_sq / in_width + eps);
        }
        for (size_t i = 0; i < out_width; ++i) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val_f = llaisys::utils::cast<float>(in_row[i]) / rms;
                val_f *= llaisys::utils::cast<float>(weight[i]);
                out_row[i] = llaisys::utils::cast<T>(val_f);
            } else {
                out_row[i] = in_row[i] / rms * weight[i];
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, llaisysDataType_t type, size_t batch_size, size_t in_width, size_t out_width) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in),
                    reinterpret_cast<float *>(weight), eps, 
                    batch_size, in_width, out_width);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in),
                    reinterpret_cast<llaisys::bf16_t *>(weight), eps, 
                    batch_size, in_width, out_width);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in),
                    reinterpret_cast<llaisys::fp16_t *>(weight), eps,
                    batch_size, in_width, out_width);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
