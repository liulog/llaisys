#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t index_num, size_t weight_width);
}
