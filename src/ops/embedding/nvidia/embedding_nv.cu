#include "embedding_nv.cuh"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, T *weight, size_t index_num, size_t weight_width) {
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < index_num) {
        int64_t word_id = index[idx];
        // Copy the corresponding row from weight to out
        for (size_t j = 0; j < weight_width; ++j) {
            out[idx * weight_width + j] = weight[word_id * weight_width + j];
        }
    }
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t index_num, size_t weight_width) {
    size_t threads_per_block = 256;
    size_t num_blocks = (index_num + threads_per_block - 1) / threads_per_block;
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        embedding_kernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<int64_t *>(index),
            reinterpret_cast<float *>(weight),
            index_num,
            weight_width);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        embedding_kernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<int64_t *>(index),
            reinterpret_cast<__nv_bfloat16 *>(weight),
            index_num,
            weight_width);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        embedding_kernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<__nv_half *>(out),
            reinterpret_cast<int64_t *>(index),
            reinterpret_cast<__nv_half *>(weight),
            index_num,
            weight_width);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
