#pragma once
#include <memory>
#include <cstdint>
#include <functional>
#include "tensor.h"
#include "tensor_node.h"

namespace dnnbasic::loss
{
	template<typename T>
	class lossData
	{
	private:
		tensor<T> gradient;
		std::shared_ptr<tensorNode<T>> leafNode;
		tensor<T> errorTensor;
		std::function<T(tensor<T>)> errorCalcMethod;

	public:	
		lossData(tensor<T> gradient, tensor<T> error, std::shared_ptr<tensorNode<T>> leafNode, const std::function<T(tensor<T>)>& errorCalcMethod);

		void backward(optimizer::optimizer* opti);

		T getError();
	};

	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual);

	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual, const uint32_t batchDim);
}
