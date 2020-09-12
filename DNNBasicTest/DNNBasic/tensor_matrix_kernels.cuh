#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "tensor_def.h"

namespace dnnbasic
{
	void tensorMatrixMul(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
}