#pragma once

#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	template <typename T>
	void tensorReLU(const tensor<T>& input, tensor<T>& output);

	template<typename T>
	void tensorReLUDerivative(const tensor<T>& derivative_activation_function, const tensor<T>& affine_input, tensor<T>& output);
}