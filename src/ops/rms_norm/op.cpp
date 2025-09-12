#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nv.cuh"
#endif

namespace llaisys::ops {
/**
 * @brief Root Mean Square Normalization (RMS Norm)
 *        Yi = Wi * Xi / sqrt(Sum(Xj^2)/n + eps)
 * 
 * @param out [batch_size, out_width]
 * @param in [batch_size, in_width]
 * @param weight [in_width]
 * @param eps float
 */
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(weight->dtype(), out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMS_Norm: all tensors must be contiguous.");

    ASSERT(out->ndim() == 2, "RMS_Norm: output tensor must be 2-dimensional.");
    ASSERT(in->ndim() == 2, "RMS_Norm: input tensor must be 2-dimensional.");
    ASSERT(weight->ndim() == 1, "RMS_Norm: weight tensor must be 2-dimensional.");

    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, weight->dtype(), in->shape()[0], in->shape()[1], out->shape()[1]);
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, weight->dtype(), in->shape()[0], in->shape()[1], out->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, weight->dtype(), in->shape()[0], in->shape()[1], out->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
