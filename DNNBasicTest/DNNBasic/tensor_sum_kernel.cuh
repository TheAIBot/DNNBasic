#pragma once

#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	template<typename T>
	void tensorSum(const tensor<T>& input, tensor<T>& output, const uint32_t sumDimIdx);
}