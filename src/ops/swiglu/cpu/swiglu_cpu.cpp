#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t seqlen, size_t intermediate_size) {    
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t j = 0; j < intermediate_size; ++j) {
            size_t index = i * intermediate_size + j;
            float gate_value, up_value;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                gate_value = llaisys::utils::cast<float>(gate[index]);
                up_value = llaisys::utils::cast<float>(up[index]);
                out[index] = llaisys::utils::cast<T>(up_value * gate_value / (1 + std::exp(-gate_value)));
            } else {
                gate_value = gate[index];
                up_value = up[index];
                out[index] = up_value * gate_value / (1 + std::exp(-gate_value));
            }
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t type, size_t seqlen, size_t intermediate_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(gate),
                    reinterpret_cast<float *>(up), seqlen, intermediate_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(gate),
                    reinterpret_cast<llaisys::bf16_t *>(up), seqlen, intermediate_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(gate),
                    reinterpret_cast<llaisys::fp16_t *>(up), seqlen, intermediate_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
