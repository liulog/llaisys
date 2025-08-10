#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
namespace llaisys::ops {
/**
 * @brief Y = x * W^T + b
 * 
 * @param out 
 * @param in 
 * @param weight 
 * @param bias 
 */
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    CHECK_SAME_DTYPE(weight->dtype(), out->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");

    ASSERT(out->ndim() == 2, "Linear: output tensor must be 2-dimensional.");
    ASSERT(in->ndim() == 2, "Linear: input tensor must be 2-dimensional.");
    ASSERT(weight->ndim() == 2, "Linear: weight tensor must be 2-dimensional.");
    ASSERT(bias->ndim() == 1, "Linear: bias tensor must be 1-dimensional.");

    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
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
