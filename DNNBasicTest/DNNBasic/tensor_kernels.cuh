#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include "span.h"
#include "tensor_def.h"

namespace dnnbasic
{
	void tensorMultiply(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
	void tensorMultiply(const float left, const tensor<float>& right, const tensor<float>& result);

	void tensorAdd(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
	void tensorAdd(const float left, const tensor<float>& right, const tensor<float>& result);

	void tensorSubtract(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
	void tensorSubtract(const float left, const tensor<float>& right, const tensor<float>& result);
}