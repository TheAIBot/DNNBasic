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
	lossData<T>::lossData(tensor<T> gradient, tensor<T> error, std::shared_ptr<tensorNode<T>> leafNode, const std::function<T(tensor<T>)>& errorCalcMethod) :
		gradient(gradient),
		errorTensor(error),
		leafNode(leafNode),
		errorCalcMethod(errorCalcMethod)
	{ }


	template<typename T>
	void lossData<T>::backward(optimizer::optimizer* opti)
	{
		this->leafNode->backward(this->gradient, opti, std::vector<activations::activationFunction<T>*>(), true);
	}

	template<typename T>
	T lossData<T>::getError()
	{
		return this->errorCalcMethod(this->errorTensor);
	}

	template class lossData<bool>;
	template class lossData<uint8_t>;
	template class lossData<uint16_t>;
	template class lossData<uint32_t>;
	template class lossData<uint64_t>;
	template class lossData<int8_t>;
	template class lossData<int16_t>;
	template class lossData<int32_t>;
	template class lossData<int64_t>;
	template class lossData<float>;
	template class lossData<double>;
}


