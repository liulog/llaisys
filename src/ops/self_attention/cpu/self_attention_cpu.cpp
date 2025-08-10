#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

/**
 * @brief Self-Attention operation
 * 
 * @tparam T 
 * @param attn_val      [seqlen, nhead, dv]         output vector
 * @param q             [seqlen, nhead, d]          query vector, hidden_size = nhead * d
 * @param k             [total_len, nkvhead, d]     key vector
 * @param v             [total_len, nkvhead, dv]    value vector
 * @param scale         scale = 1 / sqrt(d)
 * @param seqlen        sequence length (number of query tokens)
 * @param nhead         number of attention heads (query heads)
 * @param dv            dimension of value vector
 * @param d             dimension of query/key vector
 * @param total_len     total length of key/value vectors
 * @param nkvhead       number of key/value heads (usually same as nhead)
 */
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T * v, const float scale, size_t seqlen, size_t nhead, size_t dv, size_t d, size_t total_len, size_t nkvhead) {
    size_t group = nhead / nkvhead; // group > 1 means key/value heads are shared by multiple query heads
    
    std::vector<double> logits(total_len); // used for softmax calculation
    std::vector<double> weights(total_len);

    // A = Q K^T * scale      [seqlen, nhead, d] * [total_len, nkvhead, d] -> [seqlen, nhead, total_len]
    // B = causal_softmax(A)    [seqlen, nhead, total_len] -> [seqlen, nhead, total_len]
    // Attn = B V             [seqlen, nhead, total_len] * [total_len, nkvhead, dv] -> [seqlen, nhead, dv]

    for (size_t t = 0; t < seqlen; ++t) {
        for (size_t h = 0; h < nhead; ++h) {
            // Calculate the index for query head's corresponding key/value head
            size_t kv_index = (group > 0) ? (h / group) : h;
            if (kv_index >= nkvhead) kv_index = nkvhead - 1;
            
            // Query vector
            const T* q_ptr = q + (t * nhead + h) * d;
            // 1. A = scale * (Q K^T)
            // [seqlen, nhead, d] * [total_len, nkvhead, d] -> [seqlen, nhead, total_len]
            double max_logit = -std::numeric_limits<double>::infinity();
            for (size_t pos = 0; pos < total_len; ++pos) {
                // Key vector
                const T* k_ptr = k + (pos * nkvhead + kv_index) * d;
                double dot = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    double qv, kvv;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                        qv = llaisys::utils::cast<double>(q_ptr[j]);
                        kvv = llaisys::utils::cast<double>(k_ptr[j]);
                        dot += qv * kvv;
                    }
                    else {
                        qv = q_ptr[j];
                        kvv = k_ptr[j];
                        dot += qv * kvv;
                    }
                }
                double scaled = dot * scale;
                logits[pos] = scaled;
                if (scaled > max_logit && pos >= t) max_logit = scaled;
            }

            // 2. B = causal_softmax(A)
            // [seqlen, nhead, total_len] -> [seqlen, nhead, total_len]
            double sum_exp = 0.0;
            for (size_t pos = 0; pos < total_len; ++pos) {
                if (pos > t) {  // t represents current token.
                    weights[pos] = 0.0f;
                    continue;
                }
                double e = std::exp(static_cast<double>(logits[pos] - max_logit));
                weights[pos] = e;
                sum_exp += e;
            }
            double inv_sum = (sum_exp == 0.0) ? 0.0f : static_cast<double>(1.0 / sum_exp);
            for (size_t pos = 0; pos < total_len; ++pos) {
                weights[pos] = weights[pos] * inv_sum;
            }

            // 3. Attn = B V
            // [seqlen, nhead, total_len] * [total_len, nkvhead, dv] -> [seqlen, nhead, dv]
            T* out_ptr = attn_val + (t * nhead + h) * dv;
            std::vector<double> acc(dv, 0.0f);

            for (size_t pos = 0; pos < total_len; ++pos) {
                // Value vector
                const T* v_ptr = v + (pos * nkvhead + kv_index) * dv;
                double w = weights[pos];
                if (w == 0.0f) continue;
                for (size_t m = 0; m < dv; ++m) {
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        acc[m] += w * llaisys::utils::cast<double>(v_ptr[m]);
                    }
                    else {
                        acc[m] += w * static_cast<double>(v_ptr[m]);
                    }
                }
            }

            for (size_t m = 0; m < dv; ++m) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_ptr[m] = llaisys::utils::cast<T>(acc[m]);
                } else {
                    out_ptr[m] = static_cast<float>(acc[m]);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v, float scale, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t dv, size_t d, size_t total_len, size_t nkvhead) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<float *>(q),
                    reinterpret_cast<float *>(k), reinterpret_cast<float *>(v), scale,
                    seqlen, nhead, dv, d, total_len, nkvhead);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<llaisys::bf16_t *>(q),
                    reinterpret_cast<llaisys::bf16_t *>(k), reinterpret_cast<llaisys::bf16_t *>(v), scale,
                    seqlen, nhead, dv, d, total_len, nkvhead);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<llaisys::fp16_t *>(q),
                    reinterpret_cast<llaisys::fp16_t *>(k), reinterpret_cast<llaisys::fp16_t *>(v), scale,
                    seqlen, nhead, dv, d, total_len, nkvhead);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
