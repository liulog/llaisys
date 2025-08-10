#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
/**
 * @brief Rotary Position Embedding (RoPE)
 * 
 * @param out [seqlen, nhead, d]
 * @param in [seqlen, nhead, d]
 * @param pos_ids [seqlen,]
 * @param theta float
 */
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(in->dtype(), out->dtype());
    CHECK_SAME_DTYPE(pos_ids->dtype(), LLAISYS_DTYPE_I64);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    ASSERT(out->ndim() == 3, "RoPE: output tensor must be 3-dimensional.");
    ASSERT(in->ndim() == 3, "RoPE: input tensor must be 3-dimensional.");
    ASSERT(pos_ids->ndim() == 1, "RoPE: position ids tensor must be 1-dimensional.");

    // always support cpu calculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(), in->shape()[0], in->shape()[1], in->shape()[2]);
    }
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(), in->shape()[0], in->shape()[1], in->shape()[2]);
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
