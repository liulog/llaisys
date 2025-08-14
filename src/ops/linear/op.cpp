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
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(weight->dtype(), out->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    ASSERT(out->ndim() == 2, "Linear: output tensor must be 2-dimensional.");
    ASSERT(in->ndim() == 2, "Linear: input tensor must be 2-dimensional.");
    ASSERT(weight->ndim() == 2, "Linear: weight tensor must be 2-dimensional.");
    ASSERT(weight->shape()[1] == in->shape()[1], "Linear: weight tensor's second dimension must match input tensor's second dimension.");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: output tensor's second dimension must match weight tensor's first dimension.");
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: output and input tensors must have the same first dimension.");

    std::byte *bias_ptr;
    if (bias) {
        ASSERT(bias->ndim() == 1 || bias->ndim() == 0, "Linear: bias tensor must be 1-dimensional or scalar.");
        bias_ptr = bias->data();
    } else {
        bias_ptr = nullptr;
    }
    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, weight->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, weight->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
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
