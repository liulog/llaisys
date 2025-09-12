#include "argmax_nv.cuh"

#include "../../../utils.hpp"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

/**
 * @brief Due to the limitation of atomic operations on floating-point types in CUDA, only the index is stored using atomicMax.
 * 
 * @tparam T 
 * @param max_idx 
 * @param vals 
 * @param numel 
 * @return __global__ 
 */
template <typename T>
__global__ void argmax_block_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    extern __shared__ char shared_mem[];
    T *shared_vals = reinterpret_cast<T*>(shared_mem);
    int64_t *shared_idxs = reinterpret_cast<int64_t*>(&shared_vals[blockDim.x]);

    size_t tid = threadIdx.x;
    size_t threads = blockDim.x;

    T local_max;
    int64_t local_idx = -1;

    if constexpr (std::is_same_v<T, __nv_half>) {
        local_max = __float2half(-FLT_MAX);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        local_max = __float2bfloat16(-FLT_MAX);
    } else {
        local_max = -FLT_MAX;
    }

    for (size_t i = tid; i < numel; i += threads) {
        T val = vals[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    __syncthreads();

    // Block reduction to find the maximum value and its index
    for (size_t stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid] < shared_vals[tid + stride]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the result from the first thread
    if (tid == 0) {
        *max_val = shared_vals[0];
        *max_idx = shared_idxs[0];
    }
}

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val,
            std::byte *vals, llaisysDataType_t type, size_t numel) {
    const int threads_per_block = 1024;
    size_t shared_mem_size = 0;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        shared_mem_size = threads_per_block * (sizeof(float) + sizeof(int64_t));
        argmax_block_kernel<<<1, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel
        );
        break;
    case LLAISYS_DTYPE_BF16:
        shared_mem_size = threads_per_block * (sizeof(__nv_bfloat16) + sizeof(int64_t));
        argmax_block_kernel<<<1, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals),
            numel
        );
        break;
    case LLAISYS_DTYPE_F16:
        shared_mem_size = threads_per_block * (sizeof(__nv_half) + sizeof(int64_t));
        argmax_block_kernel<<<1, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_half *>(max_val),
            reinterpret_cast<const __nv_half *>(vals),
            numel
        );
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia