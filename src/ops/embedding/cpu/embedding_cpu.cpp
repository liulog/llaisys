#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, T *weight, size_t index_num, size_t weight_width) {
    for (size_t i = 0; i < index_num; ++i) {
        int64_t idx = index[i];
        std::memcpy(
            out + i * weight_width,
            weight + idx * weight_width,
            weight_width * sizeof(T)
        );
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t index_num, size_t weight_width) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<int64_t *>(index),
                    reinterpret_cast<float *>(weight), index_num, weight_width);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<int64_t *>(index),
                    reinterpret_cast<llaisys::bf16_t *>(weight), index_num, weight_width);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<int64_t *>(index),
                    reinterpret_cast<llaisys::fp16_t *>(weight), index_num, weight_width);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
