#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, llaisysDataType_t type, size_t batch_size, size_t in_width, size_t out_width);
}