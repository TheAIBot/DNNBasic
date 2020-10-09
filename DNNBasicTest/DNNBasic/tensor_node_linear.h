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
		tensor<T> outputTensor;
		const layer::linear<T>* linear;

	public:
		tensorNodeLinearLayer(tensor<T> input, tensor<T> output, const layer::linear<T>* linear) : inputNode(input.getNode()), outputTensor(output), linear(linear)
		{ }

		void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti) const override
		{
			linear->backward(estimatedLoss, opti);
		}
	};
}