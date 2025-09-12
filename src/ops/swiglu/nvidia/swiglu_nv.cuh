#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t type, size_t seqlen, size_t intermediate_size);
}