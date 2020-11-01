#pragma once
#include <memory>
#include <cstdint>
#include <functional>
#include "tensor.h"
#include "tensor_node.h"
#include "loss_data.h"

namespace dnnbasic::loss
{
	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual);

	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual, const uint32_t batchDim);
}
