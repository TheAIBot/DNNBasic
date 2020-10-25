#pragma once

#include <vector>
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

		void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti, std::vector<activations::activationFunction<T>*> actFuncs, bool isFirstLayer) const override
		{
			auto newLoss = linear->backward(estimatedLoss, opti, this->inputTensor, this->outputTensor, actFuncs, isFirstLayer);
			if (inputNode.has_value())
			{
				inputNode.value()->backward(newLoss, opti, std::vector<activations::activationFunction<T>*>(), false);
			}
		}

		virtual std::vector<tensor<T>> getTensors() const override
		{
			return { inputTensor, outputTensor };
		}
	};
}