#pragma once

#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	template <typename T>
	void tensorReLU(const tensor<T>& input, const tensor<T>& output);
}