#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

/**
 * @brief Linear layer operation: Y = X * W^T + b
 * 
 * @param out [batch_size, out_width]
 * @param in [batch_size, in_width]
 * @param weight [out_width, in_width]
 * @param bias [out_width]
 * @param batch_size 
 * @param in_width 
 * @param out_width 
 */
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t batch_size, size_t in_width, size_t out_width) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_width; ++j) {
                float sum = bias==nullptr ? 0.0f : llaisys::utils::cast<float>(bias[j]);
                for (size_t k = 0; k < in_width; ++k) {
                    sum = std::fmaf(llaisys::utils::cast<float>(in[i * in_width + k]), llaisys::utils::cast<float>(weight[j * in_width + k]), sum);
                }
                out[i * out_width + j] = llaisys::utils::cast<T>(sum);
            }
        }
    } else {
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_width; ++j) {
                float sum = bias==nullptr ? 0.0f : bias[j];
                for (size_t k = 0; k < in_width; ++k) {
                    sum = std::fmaf(in[i * in_width + k], weight[j * in_width + k], sum);
                }
                out[i * out_width + j] = sum;
            }
        }

    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_width, size_t out_width) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in),
                    reinterpret_cast<float *>(weight), bias ? reinterpret_cast<float *>(bias): nullptr, 
                    batch_size, in_width, out_width);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in),
                    reinterpret_cast<llaisys::bf16_t *>(weight), bias ? reinterpret_cast<llaisys::bf16_t *>(bias): nullptr, 
                    batch_size, in_width, out_width);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in),
                    reinterpret_cast<llaisys::fp16_t *>(weight), bias ? reinterpret_cast<llaisys::fp16_t *>(bias): nullptr, 
                    batch_size, in_width, out_width);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
