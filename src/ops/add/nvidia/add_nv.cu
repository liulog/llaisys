#include "add_nv.cuh"

#include "../../../utils.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
            c[i] = a[i] + b[i];
    }
}

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    size_t threads_per_block = 256;
    size_t num_blocks = (numel + threads_per_block - 1) / threads_per_block;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float *d_c = reinterpret_cast<float*>(c);
        const float *d_a = reinterpret_cast<const float*>(a);
        const float *d_b = reinterpret_cast<const float*>(b);
        add_kernel<float><<<num_blocks, threads_per_block>>>(d_c, d_a, d_b, numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        __nv_bfloat16 *d_c = reinterpret_cast<__nv_bfloat16*>(c);
        const __nv_bfloat16 *d_a = reinterpret_cast<const __nv_bfloat16*>(a);
        const __nv_bfloat16 *d_b = reinterpret_cast<const __nv_bfloat16*>(b);
        add_kernel<__nv_bfloat16><<<num_blocks, threads_per_block>>>(d_c, d_a, d_b, numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        __nv_half *d_c = reinterpret_cast<__nv_half*>(c);
        const __nv_half *d_a = reinterpret_cast<const __nv_half*>(a);
        const __nv_half *d_b = reinterpret_cast<const __nv_half*>(b);
        add_kernel<__nv_half><<<num_blocks, threads_per_block>>>(d_c, d_a, d_b, numel);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia