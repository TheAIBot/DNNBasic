#include "linear.h"
#include "tensor_node_linear.h"
#include "auto_graph.h"

namespace dnnbasic
{
	namespace layer
	{

		template<typename T>
		linear<T>::linear(const uint32_t inputDim, const uint32_t outputDim, const bool useBias) : weights({ inputDim,outputDim }), biases({ useBias ? outputDim : 0 })
		{
			this->useBias = useBias;
			//this->weights.makeRandom();

			if (useBias)
			{
				//this->biases.makeRandom();
			}
		}

		template<typename T>
		tensor<T> linear<T>::forward(const tensor<T>& x) const
		{
			autoGraph::scopeLevelDisableAutoGrad t;

			tensor<T> output = this->useBias ? 
				this->weights.matMul(x) + this->biases :
				this->weights.matMul(x);
			output.setNode(new tensorNodeLinearLayer<T>(x, output, this));

			return output;
		}
		template<typename T>
		void linear<T>::backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti) const
		{
			autoGraph::scopeLevelDisableAutoGrad t;
		}

		template class linear<bool>;
		template class linear<uint8_t>;
		template class linear<uint16_t>;
		template class linear<uint32_t>;
		template class linear<uint64_t>;
		template class linear<int8_t>;
		template class linear<int16_t>;
		template class linear<int32_t>;
		template class linear<int64_t>;
		template class linear<float>;
		template class linear<double>;
	}
}