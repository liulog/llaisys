#include "rms_norm_nv.cuh"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * @brief Root Mean Square Normalization (RMS Norm)
 *       Yi = Wi * Xi / sqrt(Sum(Xj^2)/n + eps
 * 
 * @tparam T 
 * @param out [batch_size, out_width]
 * @param in [batch_size, in_width]
 * @param weight [in_width]
 * @param eps float
 * @param batch_size 
 * @param in_width 
 * @param out_width 
 */
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                const float eps, size_t batch_size,
                                size_t in_width, size_t out_width) {
    // blockDim.x <= 256
    size_t batch_id = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t data_per_thread = (in_width + blockDim.x - 1) / blockDim.x;

    // Step 1: local sum of squares
    float local_sum = 0.0f;
    for (size_t i = 0; i < data_per_thread; i++) {
        size_t idx = tid + i * blockDim.x;
        if (idx < in_width) {
            float val = static_cast<float>(in[batch_id * in_width + idx]);
            local_sum += val * val;
        }
    }

    // Step 2: block reduce
    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float row_rms;
    if (tid == 0) {
        row_rms = rsqrtf(shared_sum[0] / in_width + eps);
    }
    __syncthreads();

    // Step 3: apply RMSNorm
    for (size_t i = 0; i < data_per_thread; i++) {
        size_t idx = tid + i * blockDim.x;
        if (idx < out_width) {
            float val = static_cast<float>(in[batch_id * in_width + idx]);
            float w = static_cast<float>(weight[idx]);
            out[batch_id * out_width + idx] = static_cast<T>(val * row_rms * w);
        }
    }
}


namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, llaisysDataType_t type, size_t batch_size, size_t in_width, size_t out_width) {
    ASSERT(in_width == out_width, "in_width must be equal to out_width");

    size_t threads_per_block = 256;
    switch (type) {
    case LLAISYS_DTYPE_F32:{
        rms_norm_kernel<<<batch_size, threads_per_block>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            eps,
            batch_size,
            in_width,
            out_width
        );
        break;
    }
    case LLAISYS_DTYPE_BF16:{
        rms_norm_kernel<<<batch_size, threads_per_block>>>(
            reinterpret_cast<__nv_bfloat16*>(out),
            reinterpret_cast<const __nv_bfloat16*>(in),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            eps,
            batch_size,
            in_width,
            out_width
        );
        break;
    }
    case LLAISYS_DTYPE_F16: {
        rms_norm_kernel<<<batch_size, threads_per_block>>>(
            reinterpret_cast<__half*>(out),
            reinterpret_cast<const __half*>(in),
            reinterpret_cast<const __half*>(weight),
            eps,
            batch_size,
            in_width,
            out_width
        );
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
