#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_width, size_t out_width);
}