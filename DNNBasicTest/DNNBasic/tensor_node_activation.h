#pragma once

#include <vector>
#include "tensor.h"
#include "optional.h"
#include "FBPropagation.h"
#include "activation.h"

namespace dnnbasic
{
	template<typename T>
	class tensorNodeActivation : public tensorNode<T>
	{
	private:
		optional<std::shared_ptr<tensorNode<T>>> node;
		tensor<T> inputTensor;
		activations::activationFunction<T>* activation;

	public:
		tensorNodeActivation(tensor<T> inputTensor, activations::activationFunction<T>* activation) : node(inputTensor.getNode()), inputTensor(input), activation(activation)
		{ }

		void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti, std::vector<activations::activationFunction<T>*> actFuncs, bool isFirstLayer) const override
		{
			actFuncs->push_back(activation);
			if (node.hasValue())
			{
				node.value()->backward(estimatedLoss, opti, actFuncs, isFirstLayer);
			}
		}

		virtual std::vector<tensor<T>> getTensors() const override
		{
			return { inputTensor };
		}
	};
}