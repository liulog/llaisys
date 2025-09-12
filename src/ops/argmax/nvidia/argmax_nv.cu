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
__global__ void get_maxidx_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel, int *lock) {
    // Shared memory
    extern __shared__ char shared_mem[];  // Shared memory for values and indices
    T *shared_vals = (T*)shared_mem;  
    int64_t *shared_idxs = (int64_t*)(&shared_vals[blockDim.x]);  // Ensures that shared_idxs follows shared_vals
    
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread index within the block
    size_t tid = threadIdx.x;
    // Initial stride for reduction
    size_t half = blockDim.x / 2;

    // Read data into shared memory
    if (idx < numel) {
        shared_vals[tid] = vals[idx]; // Global data value
        shared_idxs[tid] = idx; // Global data index
    } else {
        if constexpr (std::is_same_v<T, __nv_half>) {
            shared_vals[tid] = __float2half(-FLT_MAX);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            shared_vals[tid] = __float2bfloat16(-FLT_MAX);
        } else {
            shared_vals[tid] = -FLT_MAX;
        }
        shared_idxs[tid] = -1;
    }
    __syncthreads();

    // Parallel reduction within the block to find the maximum value
    for (size_t stride = half; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (shared_vals[tid] < shared_vals[tid + stride]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];  // Update the index of max value
            }
        }
        __syncthreads();
    }

    // The thread with index 0 in each block stores the maximum value and index
    if (tid == 0) {
        // Due to float and fp16/bf16 atomic operation limitations, only the index is stored.
        // And we need change val and index atomically.
        while (atomicCAS(lock, 0, 1) != 0) {
            // Busy-wait until the lock is acquired
        }
        // Compare and update the max value and index
        int idx = shared_idxs[0];
        T value = shared_vals[0];
        if (value > *max_val) {
            *max_val = value;
            *max_idx = idx;
        }
        atomicExch(lock, 0);
    }
}

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, std::byte *vals, llaisysDataType_t type, size_t numel) {
    size_t threads_per_block = 1024;
    size_t num_blocks = (numel + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = sizeof(int64_t) * threads_per_block;
    
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        shared_mem_size += sizeof(float) * threads_per_block;
        // Used for atomic operations
        void *d_lock;
        cudaError_t err = cudaMalloc(&d_lock, sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        cudaMemset(d_lock, 0, sizeof(int));
        get_maxidx_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel,
            (int *)d_lock
        );
        cudaFree(d_lock);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
            // Used for atomic operations
    void *d_lock;
        cudaError_t err = cudaMalloc(&d_lock, sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        cudaMemset(d_lock, 0, sizeof(int));
        shared_mem_size += sizeof(__nv_bfloat16) * threads_per_block;
        get_maxidx_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals),
            numel,
            (int *)d_lock
        );
        cudaFree(d_lock);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        // Used for atomic operations
        void *d_lock;
        cudaError_t err = cudaMalloc(&d_lock, sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        cudaMemset(d_lock, 0, sizeof(int));
        shared_mem_size += sizeof(__nv_half) * threads_per_block;
        get_maxidx_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_half *>(max_val),
            reinterpret_cast<const __nv_half *>(vals),
            numel,
            (int *)d_lock
        );
        cudaFree(d_lock);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

}
} // namespace llaisys::ops::nvidia