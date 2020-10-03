#pragma once
#include "Tensor.h"
#include "Layer.h"


namespace dnnbasic 
{
	template<typename T>
	class tensorNodeNoGrad : public tensorNode<T>
	{
		layer::linear<T>* layer;

		tensorNodeNoGrad(tensor<T> input, tensor<T> output, )

		void backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{
			layer.backward(estimatedLoss, functionOut);
		}
	};
}