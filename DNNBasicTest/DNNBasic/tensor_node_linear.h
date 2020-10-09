#pragma once
#include "tensor.h"
#include "optional.h"
#include "FBPropagation.h"
#include "linear.h"

namespace dnnbasic 
{
	template<typename T>
	class tensorNodeLinearLayer : public tensorNode<T>
	{
		optional<std::shared_ptr<tensorNode<T>>> inputNode;
		tensor<T> outputTensor;
		layer::linear<T>* linear;

		tensorNodeLinearLayer(tensor<T> input, tensor<T> output, layer::linear<T>* linear)
		{
			this->inputNode = input.getNode();
			this->outputTensor = output;
			this->linear = linear;
		}

		void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti) const override
		{
			linear->backward(estimatedLoss, opti);
		}
	};
}