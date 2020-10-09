#include "linear.h"
#include "tensor_node_linear.h"
#include "auto_graph.h"

namespace dnnbasic
{
	namespace layer
	{

		template<typename T>
		linear<T>::linear(int inputDim, int outputDim, bool useBias)
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

		template<typename T>
		tensor<T> linear<T>::forward(const tensor<T>& x) const
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
		template<typename T>
		void linear<T>::backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti) const
		{
			autoGraph::scopeLevelDisableAutoGrad t;
		}
	}
}