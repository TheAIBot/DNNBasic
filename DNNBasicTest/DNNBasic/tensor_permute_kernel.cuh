#pragma once

#include <vector>
#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	template <typename T>
	void tensorPermute(const tensor<T>& input, const tensor<T>& output, const std::vector<uint32_t>& dims);
}