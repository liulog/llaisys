#include "swiglu_nv.cuh"

#include "../../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t seqlen, size_t intermediate_size) {
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < seqlen * intermediate_size) {
        T gate_val = gate[idx];
        T up_val = up[idx];
        T sigmoid = static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(exp(-static_cast<float>(gate_val))));
        out[idx] = up_val * gate_val * sigmoid;
    }
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t type, size_t seqlen, size_t intermediate_size) {
    size_t threads_per_block = 1024;
    size_t blocks = (seqlen * intermediate_size + threads_per_block - 1) / threads_per_block;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        swiglu_kernel<<<blocks, threads_per_block>>>(reinterpret_cast<float*>(out), 
            reinterpret_cast<const float*>(gate), reinterpret_cast<const float*>(up),
            seqlen, intermediate_size);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        swiglu_kernel<<<blocks, threads_per_block>>>(reinterpret_cast<__nv_bfloat16*>(out), 
            reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
            seqlen, intermediate_size);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        swiglu_kernel<<<blocks, threads_per_block>>>(reinterpret_cast<__nv_half*>(out),
            reinterpret_cast<const __nv_half*>(gate), reinterpret_cast<const __nv_half*>(up),
            seqlen, intermediate_size);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
