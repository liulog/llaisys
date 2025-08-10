#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

/**
 * @brief Rotary Position Embedding (RoPE)
 *        For some fp32 test case, we need use double for precision 
 * @tparam T 
 * @param out [seqlen, nhead, d]
 * @param in [seqlen, nhead, d], Q or K
 * @param pos_ids [seqlen,]
 * @param theta float
 * @param seqlen seqlen
 * @param nhead nhead
 * @param d d = hidden_size / nhead
 */
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, const float theta, size_t seqlen, size_t nhead, size_t d) {
    ASSERT(d % 2 == 0, "RoPE: Head dimension must be even for RoPE.");
    // Xi = (Xj, Xk), i in n, j in a, k in b, where a = b = d/2
    size_t d_half = d / 2;
    // Precompute inv_freq[j] = 1 / theta^(2j/d), avoiding repeated calculations
    std::vector<double> inv_freq(d_half);
    for (size_t i = 0; i < d_half; ++i) {
        inv_freq[i] = 1.0f / std::pow((double)theta, (double)i / d_half);
    }
    for (size_t t = 0; t < seqlen; ++t) {
        int64_t pos = pos_ids[t];
        for (size_t h = 0; h < nhead; ++h) {
            // For each token, handle each head separately
            const T* in_ptr = in + (t * nhead + h) * d;
            T* out_ptr = out + (t * nhead + h) * d;
            // Rope core code
            for (size_t i = 0; i < d_half; ++i) {
                // Calculate the angle for the current position and frequency
                double phi = pos * static_cast<double>(inv_freq[i]);
                double cos_phi = std::cos(phi);
                double sin_phi = std::sin(phi);
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                    double old_a = llaisys::utils::cast<float>(in_ptr[i]);
                    double old_b = llaisys::utils::cast<float>(in_ptr[i + d_half]);
                    double new_a = old_a * cos_phi - old_b * sin_phi;
                    double new_b = old_b * cos_phi + old_a * sin_phi;
                    out_ptr[i] = llaisys::utils::cast<T>(new_a);
                    out_ptr[i + d_half] = llaisys::utils::cast<T>(new_b);
                } else {
                    double old_a = in_ptr[i];
                    double old_b = in_ptr[i + d_half];
                    double new_a = old_a * cos_phi - old_b * sin_phi;
                    double new_b = old_b * cos_phi + old_a * sin_phi;
                    out_ptr[i] = new_a;
                    out_ptr[i + d_half] = new_b;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, std::byte *in, std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in),
                    reinterpret_cast<int64_t *>(pos_ids), theta, 
                    seqlen, nhead, n);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in),
                    reinterpret_cast<int64_t *>(pos_ids), theta, 
                    seqlen, nhead, n);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in),
                    reinterpret_cast<int64_t *>(pos_ids), theta,
                    seqlen, nhead, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
