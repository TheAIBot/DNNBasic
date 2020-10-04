#pragma once
#include "tensor.h"
#include "Layer.h"
#include "optional.h"

namespace dnnbasic 
{
	template<typename T>
	class tensorNodeLinearLayer : public tensorNode<T>
	{
		optional<std::shared_ptr<tensorNode<T>>> inputNode;
		tensor<T> outputTensor;
		layer::linear<T>* layer;

		tensorNodeLinearLayer(tensor<T> input, tensor<T> output, layer::linear* layer)
		{
			this->inputNode = input.getNode();
			this->outputTensor = output;
			this->layer = layer;
		}

		void backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{
			layer->backward(estimatedLoss, functionOut);
		}
	};
}