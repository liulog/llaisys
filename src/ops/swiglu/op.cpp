#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
/**
 * @brief SwiGLU activation function
 * 
 * @param out [seqlen, intermediate_size]
 * @param gate [seqlen, intermediate_size]
 * @param up [seqlen, intermediate_size]
 */
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");

    ASSERT(out->ndim() == 2, "SwiGLU: out tensor must be 2-dimensional.");
    ASSERT(gate->ndim() == 2, "SwiGLU: gate tensor must be 2-dimensional.");
    ASSERT(up->ndim() == 2, "SwiGLU: up tensor must be 2-dimensional.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->shape()[0], out->shape()[1]);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->shape()[0], out->shape()[1]);
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
