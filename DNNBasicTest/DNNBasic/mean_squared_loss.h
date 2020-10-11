#pragma once
#include <memory>
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

	public:
		T error; 
	
		lossData(tensor<T> gradient, T error, std::shared_ptr<tensorNode<T>> leafNode);

		void backward(optimizer::optimizer* opti);

	};

	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual);
}
