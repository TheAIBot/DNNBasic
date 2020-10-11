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
	private:
		optional<std::shared_ptr<tensorNode<T>>> inputNode;
		tensor<T> inputTensor;
		tensor<T> outputTensor;
		layer::linear<T>* linear;

	public:
		tensorNodeLinearLayer(tensor<T> input, tensor<T> output, layer::linear<T>* linear) : inputNode(input.getNode()), inputTensor(input), outputTensor(output), linear(linear)
		{ }

		void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti) const override
		{
			auto newLoss = linear->backward(estimatedLoss, opti, this->inputTensor);
			if (inputNode.has_value())
			{
				inputNode.value()->backward(newLoss, opti);
			}
		}
	};
}