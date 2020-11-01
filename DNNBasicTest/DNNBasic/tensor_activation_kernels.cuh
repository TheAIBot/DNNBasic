#pragma once

#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	template <typename T>
	void tensorReLU(const tensor<T>& input, tensor<T>& output);

	template<typename T>
	void tensorReLUDerivative(const tensor<T>& input, tensor<T>& output);
}