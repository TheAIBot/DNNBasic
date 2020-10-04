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
		bool useBias;

	public:
		linear(int inputDim, int outputDim, bool useBias)
		{
			this->useBias = useBias;
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
			autoGraph::scopeLevelDisableAutoGrad t;

			tensor<T> output;
			if (useBias)
			{
				output = this->weights.matMul(x) + this->biases;
			}
			else
			{
				output = this->weights.matMul(x);
			}
			output.setNode(new tensorNodeLinearLayer<T>(x, output, this));

			return output;
		}
		void backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{

		}

	private:

	};
}