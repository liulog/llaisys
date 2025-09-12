#include "linear_nv.cuh"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define TILE_SIZE 32

template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias, 
                              size_t batch_size, size_t in_width, size_t out_width) {
    // Output: [batch_size, out_width]
    //           dimx                                    dimy
    // GridDim: (out_width + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE)
    // BlockDim: (TILE_SIZE, TILE_SIZE)

    // ------------->
    // |            x
    // |    out
    // |
    // v y

    // Global data index
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculation:
    // [batch_size, in_width] x [out_width, in_width]^T = [batch_size, out_width]

    __shared__ T shared_in[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_weight[TILE_SIZE][TILE_SIZE];

    float out_val = 0.0f;

    // in_width TILES
    const int num_tiles = (in_width + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // Load input to shared memory
        const size_t in_col = t * TILE_SIZE + threadIdx.x;
        if (row < batch_size && in_col < in_width) {
            shared_in[threadIdx.y][threadIdx.x] = in[row * in_width + in_col];  // in[row][in_col]
        } else {
            shared_in[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }

        // Load weight to shared memory (transposed access)
        const size_t weight_row = t * TILE_SIZE + threadIdx.y;
        if (col < out_width && weight_row < in_width) {
            shared_weight[threadIdx.y][threadIdx.x] = weight[col * in_width + weight_row];  // weight[col][weight_row] (transposed)
        } else {
            shared_weight[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();

        // Calculate in TILE
        // shared_in and shared_weight are matrix
        for (int k = 0; k < TILE_SIZE; ++k) {
            if constexpr (std::is_same_v<T, __nv_half>) {
                out_val += __half2float(shared_in[threadIdx.y][k]) * __half2float(shared_weight[k][threadIdx.x]);
            } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                out_val += __bfloat162float(shared_in[threadIdx.y][k]) * __bfloat162float(shared_weight[k][threadIdx.x]);
            } else {
                out_val += shared_in[threadIdx.y][k] * shared_weight[k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < batch_size && col < out_width) {
        const T bias_val = bias == nullptr ? T(0) : static_cast<T>(bias[col]);
        if constexpr (std::is_same_v<T, __nv_half>) {
            out[row * out_width + col] = __float2half(out_val) + bias_val;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out[row * out_width + col] = __float2bfloat16(out_val) + bias_val;
        } else {
            out[row * out_width + col] = out_val + bias_val;
        }
    }
}

namespace llaisys::ops::nvidia {
/**
 * @brief 
 * 
 * @param out [batch_size, out_width]
 * @param in [batch_size, in_width]
 * @param weight [out_width, in_width]
 * @param bias [out_width] or nullptr
 * @param type 
 * @param batch_size 
 * @param in_width 
 * @param out_width 
 */
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_width, size_t out_width) {
    
    // Grid and Block dimensions
    dim3 grid((out_width + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        linear_kernel<<<grid, block>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<float *>(in),
            reinterpret_cast<float *>(weight),
            bias ? reinterpret_cast<float *>(bias) : nullptr,
            batch_size,
            in_width,
            out_width);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        linear_kernel<<<grid, block>>>(
            reinterpret_cast<__nv_half *>(out),
            reinterpret_cast<__nv_half *>(in),
            reinterpret_cast<__nv_half *>(weight),
            bias ? reinterpret_cast<__nv_half *>(bias) : nullptr,
            batch_size,
            in_width,
            out_width);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        linear_kernel<<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<__nv_bfloat16 *>(in),
            reinterpret_cast<__nv_bfloat16 *>(weight),
            bias ? reinterpret_cast<__nv_bfloat16 *>(bias) : nullptr,
            batch_size,
            in_width,
            out_width);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia