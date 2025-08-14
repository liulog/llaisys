#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
/**
 * @brief Self-Attention operation
 *        A = Q K^T * scale             [seqlen, nhead, total_len]
 *        Attn = causal_softmax(A) V    [seqlen, nhead, dv]
 * 
 * @param attn_val [seqlen, nhead, dv]
 * @param q [seqlen, nhead, d]
 * @param k [total_len, nkvhead, d]
 * @param v [total_len, nkvhead, dv]
 * @param scale float
 */
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self-Attention: all tensors must be contiguous.");

    ASSERT(attn_val->ndim() == 3, "Self-Attention: attn_val tensor must be 3-dimensional.");
    ASSERT(q->ndim() == 3, "Self-Attention: q tensor must be 3-dimensional.");
    ASSERT(k->ndim() == 3, "Self-Attention: k tensor must be 3-dimensional.");
    ASSERT(v->ndim() == 3, "Self-Attention: v tensor must be 3-dimensional.");
    // seqlen
    ASSERT(attn_val->shape()[0] == q->shape()[0], "Self-Attention: seqlen must match between attn_val and q.");
    // nhead
    ASSERT(attn_val->shape()[1] == q->shape()[1], "Self-Attention: nhead must match between attn_val and q.");
    // total_len
    ASSERT(k->shape()[0] == v->shape()[0], "Self-Attention: total_len must match between k and v.");
    // nkvhead
    ASSERT(k->shape()[1] == v->shape()[1], "Self-Attention: nkvhead must match between k and v.");
    // dv
    ASSERT(attn_val->shape()[2] == v->shape()[2], "Self-Attention: dv must match between attn_val and v.");
    // d
    ASSERT(q->shape()[2] == k->shape()[2], "Self-Attention: d must match between q and k.");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->shape()[0], attn_val->shape()[1], attn_val->shape()[2], q->shape()[2], k->shape()[0], k->shape()[1]);
    }
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->shape()[0], attn_val->shape()[1], attn_val->shape()[2], q->shape()[2], k->shape()[0], k->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
