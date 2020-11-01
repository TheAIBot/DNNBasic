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
	lossData<T> crossEntropyLoss(tensor<T> expected, tensor<T> actual);
}
