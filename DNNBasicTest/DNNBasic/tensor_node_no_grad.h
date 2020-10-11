#pragma once
#include <vector>
#include "Tensor.h"
#include "optional.h"
namespace dnnbasic
{
	template<typename T>
	class tensorNodeNoGrad : public tensorNode<T>
	{
		std::vector<std::shared_ptr<tensorNode <T>>> nodes;

	public:
		tensorNodeNoGrad(std::vector<tensor<T>> tensorNodePtrs)
		{
			for (size_t i = 0; i < tensorNodePtrs.size(); i++)
			{
				auto tensorNode = tensorNodePtrs[i].getNode();
				if (tensorNode.has_value())
				{
					nodes.push_back(tensorNode.value());
				}
			}
		}

		void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti) const override
		{
			for (size_t i = 0; i < nodes.size(); i++)
			{
				nodes[i]->backward(estimatedLoss, opti);
			}
		}
	};
}