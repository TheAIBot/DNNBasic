#include "linear.h"
#include "tensor_node_linear.h"
#include "auto_graph.h"

#include <iostream>

namespace dnnbasic
{
	namespace layer
	{

		template<typename T>
		linear<T>::linear(const uint32_t inputDim, const uint32_t outputDim, const bool useBias) : weights({ outputDim, inputDim }), biases({ useBias ? outputDim : 1 })
		{
			this->useBias = useBias;
			//this->weights.makeRandom();

			if (useBias)
			{
				//this->biases.makeRandom();
			}
		}

		template<typename T>
		tensor<T> linear<T>::forward(const tensor<T>& x)
		{
			autoGraph::scopeLevelDisableAutoGraph t;

			tensor<T> output = this->useBias ? 
				this->weights.matMul(x) + this->biases :
				this->weights.matMul(x);
			output.setNode(new tensorNodeLinearLayer<T>(x, output, this));

			return output;
		}
		template<typename T>
		tensor<T> linear<T>::backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti, const tensor<T>& input)
		{
			autoGraph::scopeLevelDisableAutoGraph t;

			// error for layer L
			const tensor<T> newLoss = this->weights.permute({ 1, 0 }).matMul(estimatedLoss);

			// Partial derivative cost for weight
			opti->updateWeights(this->weights, estimatedLoss * input);

			// Partial derivative cost for bias
			if (this->useBias)
			{
				opti->updateWeights(this->biases, estimatedLoss);
			}

			return newLoss;
		}

		//template class linear<bool>;
		//template class linear<uint8_t>;
		//template class linear<uint16_t>;
		//template class linear<uint32_t>;
		//template class linear<uint64_t>;
		//template class linear<int8_t>;
		//template class linear<int16_t>;
		//template class linear<int32_t>;
		//template class linear<int64_t>;
		template class linear<float>;
		//template class linear<double>;
	}
}