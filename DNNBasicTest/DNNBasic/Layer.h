#pragma once
#include "FBPropagation.h"
#include "tensor.h"
#include "tensor_node_linear.h"
#include "auto_graph.h"

namespace dnnbasic::layer
{
	template<typename T>
	class linear : fbpropagation<T>
	{
	private:
		tensor<T> weights;
		tensor<T> biases;

	public:
		linear(int inputDim, int outputDim, bool useBias)
		{
			this->weights = tensor<T>({ inputDim,outputDim });
			this->weights.makeRandom();

			if (useBias)
			{
				this->biases = tensor<T>({ outputDim });
				this->biases.makeRandom();
			}
		}


		tensor<T> forward(const tensor<T>& x) const override
		{
			const bool oldMakeGraph = autoGraph::makeGraph;
			autoGraph::makeGraph = false;

			auto output = this->weights.matMul(x) + this->biases;

			autoGraph::makeGraph = oldMakeGraph;

			return output;
		}
		void backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{

		}

	private:

	};
}