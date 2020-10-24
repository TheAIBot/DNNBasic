#pragma once
#include <vector>
#include "tensor.h"
#include "optional.h"
namespace dnnbasic
{
	template<typename T>
	class tensorNodeNoGrad : public tensorNode<T>
	{
		std::vector<std::shared_ptr<tensorNode<T>>> nodes;
		std::vector<tensor<T>> tensors;

	public:
		tensorNodeNoGrad(std::vector<tensor<T>> tensors)
		{
			for (size_t i = 0; i < tensors.size(); i++)
			{
				this->tensors.push_back(tensors[i]);
				auto tensorNode = tensors[i].getNode();
				if (tensorNode.has_value())
				{
					nodes.push_back(tensorNode.value());
				}
			}
		}

		void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti, std::vector<activations::activationFunction<T>*> actFuncs, bool isFirstLayer) const override
		{
			for (size_t i = 0; i < nodes.size(); i++)
			{
				nodes[i]->backward(estimatedLoss, opti, actFuncs, isFirstLayer);
			}
		}

		virtual std::vector<tensor<T>> getTensors() const override
		{
			return this->tensors;
		}
	};
}