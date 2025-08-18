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
    // Note: group > 1 means: one kvhead are shared by group query heads
    ASSERT(nhead >= nkvhead && nhead % nkvhead == 0, "Here: nhead >= nkvhead && nhead %%nkvhead == 0");
    size_t group = nhead / nkvhead;
    
    std::vector<float> logits(total_len); // used for softmax calculation
    std::vector<float> weights(total_len); // temp vector for QK^T*scale's result

    // A = Q K^T * scale      [seqlen, nhead, d] * [total_len, nkvhead, d] -> [seqlen, nhead, total_len]
    // B = causal_softmax(A)    [seqlen, nhead, total_len] -> [seqlen, nhead, total_len]
    // Attn = B V             [seqlen, nhead, total_len] * [total_len, nkvhead, dv] -> [seqlen, nhead, dv]

    // total_len = past_len + seqlen
    // If kv_cache is used, past_len is the length of cached tokens
    // If kv_cache is not used, past_len = 0
    size_t past_len = total_len - seqlen;

    for (size_t t = 0; t < seqlen; ++t) {
        for (size_t h = 0; h < nhead; ++h) {
            // Calculate the index for query head's corresponding key/value head
            size_t kv_index = (group > 0) ? (h / group) : h;
            
            // Query vector
            const T* q_ptr = q + (t * nhead + h) * d;
            // 1. A = scale * (Q K^T)
            // [seqlen, nhead, d] * [total_len, nkvhead, d] -> [seqlen, nhead, total_len]
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t pos = 0; pos < total_len; ++pos) {
                // Key vector
                const T* k_ptr = k + (pos * nkvhead + kv_index) * d;
                float dot = 0.0f;
                // Matrix multiplication: Q * K^T
                for (size_t j = 0; j < d; ++j) {
                    float qv, kvv;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                        qv = llaisys::utils::cast<float>(q_ptr[j]);  // [t, h, j]
                        kvv = llaisys::utils::cast<float>(k_ptr[j]); // [pos, kv_index, j]
                        dot += qv * kvv;
                    }
                    else {
                        qv = q_ptr[j];
                        kvv = k_ptr[j];
                        dot += qv * kvv;
                    }
                }
                float scaled = dot * scale;
                if (t + past_len >= pos) {  // Upper triangular part is not used in causal attention
                    logits[pos] = scaled; // Keep the logits for current and past tokens
                    max_logit = std::max(scaled, max_logit);
                } else {
                    logits[pos] = -std::numeric_limits<float>::infinity(); // Masking future tokens
                    weights[pos] = -std::numeric_limits<float>::infinity(); // Initialize weights for future tokens
                }
            }

            // 2. B = causal_softmax(A)
            // [seqlen, nhead, total_len] -> [seqlen, nhead, total_len]
            float sum_exp = 0.0;
            for (size_t pos = 0; pos < total_len && t + past_len >= pos; ++pos) {
                float e = std::exp(static_cast<float>(logits[pos] - max_logit));
                weights[pos] = e;
                sum_exp += e;
            }
            ASSERT(sum_exp > 0.0, "Sum of exponentials should be greater than zero.");
            float inv_sum = static_cast<float>(1.0 / (sum_exp + 1e-6));
            for (size_t pos = 0; pos < total_len && t + past_len >= pos; ++pos) {
                weights[pos] = weights[pos] * inv_sum;
            }

            // 3. Attn = B V
            // [seqlen, nhead, total_len] * [total_len, nkvhead, dv] -> [seqlen, nhead, dv]
            T* out_ptr = attn_val + (t * nhead + h) * dv;
            std::vector<float> acc(dv, 0.0f);

            for (size_t pos = 0; pos < total_len && t + past_len >= pos; ++pos) {
                // Value vector
                const T* v_ptr = v + (pos * nkvhead + kv_index) * dv;
                float w = weights[pos];
                for (size_t m = 0; m < dv; ++m) {
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        acc[m] += w * llaisys::utils::cast<float>(v_ptr[m]); // [pos, kv_index, m]
                    }
                    else {
                        acc[m] += w * static_cast<float>(v_ptr[m]);
                    }
                }
            }

            // 4. Store the result in attn_val
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
