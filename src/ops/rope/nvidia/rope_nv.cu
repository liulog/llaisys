#include "rope_nv.cuh"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * @brief Rotary Position Embedding (RoPE)
 *        For some fp32 test case, we need use double for precision 
 * @tparam T 
 * @param out [seqlen, nhead, d]
 * @param in [seqlen, nhead, d], Q or K
 * @param pos_ids [seqlen,]
 * @param theta float
 * @param seqlen seqlen
 * @param nhead nhead
 * @param d d = hidden_size / nhead
 */
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, const float theta, size_t seqlen, size_t nhead, size_t d) {
    // Global thread index
    size_t tx = threadIdx.x;
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t seq_index = blockIdx.y / nhead;
    size_t half_d = d / 2;

    __shared__ int64_t pos_id;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        pos_id = pos_ids[seq_index];
    }
    __syncthreads();

    // Per thread handle one pair
    size_t idx = bx * blockDim.x + tx;
    if (idx < half_d) {
        double freq = 1.0 / pow((float)theta, (2.0 * (float)idx) / (float)d);
        double angle = pos_id * freq;
        float cos_angle = cos(angle);
        float sin_angle = sin(angle);
        // Important: Thread related data index, Global data index
        size_t base_index = by * d + idx;
        float in_val_0 = static_cast<float>(in[base_index]);
        float in_val_1 = static_cast<float>(in[base_index + half_d]);
        out[base_index] = static_cast<T>(in_val_0 * cos_angle - in_val_1 * sin_angle);
        out[base_index + half_d] = static_cast<T>(in_val_0 * sin_angle + in_val_1 * cos_angle);
    }
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, std::byte *in, std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t n) {
    ASSERT(n % 2 == 0, "d must be even for RoPE");
    size_t threads_per_block = 512;
    dim3 grid_dim((n / 2 + threads_per_block - 1) / threads_per_block, seqlen * nhead);
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        rope_kernel<<<grid_dim, threads_per_block>>>((float *)out, (const float *)in, (const int64_t *)pos_ids, theta, seqlen, nhead, n);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        rope_kernel<<<grid_dim, threads_per_block>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)in, (const int64_t *)pos_ids, theta, seqlen, nhead, n);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        rope_kernel<<<grid_dim, threads_per_block>>>((__half *)out, (const __half *)in, (const int64_t *)pos_ids, theta, seqlen, nhead, n);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
