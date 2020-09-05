#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include "span.h"
#include "tensor_def.h"

namespace dnnbasic
{
	void tensorMultiply(tensor<float>& left, tensor<float>& right, tensor<float>& result);
}