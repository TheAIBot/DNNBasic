#pragma once

#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	template<typename T>
	void tensorLog(const tensor<T>& input, tensor<T>& output);
}