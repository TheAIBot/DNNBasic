#include <vector>
#include "linear.h"
#include "tensor_node_linear.h"
#include "auto_graph.h"
#include "activation_function.h"

namespace dnnbasic
{
	namespace layer
	{

		template<typename T>
		linear<T>::linear(const uint32_t inputDim, const uint32_t outputDim, const bool useBias) : 
			weights(tensor<T>::random({ inputDim, outputDim }, - std::sqrt(1.0f / inputDim), std::sqrt(1.0f / inputDim))),
			biases(tensor<T>::random({ useBias ? outputDim : 1 }, - std::sqrt(1.0f / inputDim), std::sqrt(1.0f / inputDim))),
			useBias(useBias),
			inputSize(inputDim),
			outputSize(outputDim)
		{ }

		template<typename T>
		tensor<T> linear<T>::forward(const tensor<T>& x)
		{
			autoGraph::scopeLevelDisableAutoGraph t;

			tensor<T> output = this->useBias ? 
				x.matMul(this->weights) + this->biases :
				x.matMul(this->weights);
			autoGraph::forceMakeGraph(output, std::function<tensorNode<T>* ()>([&]() {return new tensorNodeLinearLayer<T>(x, output, this); }));

			return output;
		}

		template<typename T>
		tensor<T> linear<T>::backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti, const tensor<T>& input, const tensor<T>& output, std::vector<activations::activationFunction<T>*> actFuncs, bool isFirstLayer)
		{
			autoGraph::scopeLevelDisableAutoGraph t;


			// compute derivative of activation function using output
			tensor<T> newDerivative = actFuncs.size() == 0 ? tensor<T>(std::vector<uint32_t>({ 1 }), std::vector<T>({ 1 })) : actFuncs.back()->derivative(output.reshape(estimatedLoss.getDimension(0), estimatedLoss.getDimension(1)));
			if (actFuncs.size()>1)
			{
				tensor<T> forwardDerivative = actFuncs.back()->forward(output);
				for (int i = actFuncs.size() - 2; i >= 0; i--)
				{
					newDerivative = newDerivative * actFuncs[i]->derivative(forwardDerivative);
					forwardDerivative = actFuncs[i]->forward(forwardDerivative);
				}
			}

			const tensor<T> transposedInput = input.transpose(input.getDimensions().size() - 1, input.getDimensions().size() - 2);

			const tensor<T> batchWeightGradient = transposedInput.matMul(newLoss);

			auto grad = newLoss.matMul(this->weights.permute({ 1, 0 }));

			// Partial derivative cost for weight
			opti->updateWeights(this->weights, batchWeightGradient);

			// Partial derivative cost for bias
			if (this->useBias)
			{
				opti->updateWeights(this->biases, newLoss.sum(0)/((T)newLoss.getDimension(0)));
			}

			return grad;
		}

		template<typename T>
		uint32_t linear<T>::getInputSize() const
		{
			return this->inputSize;
		}
		template<typename T>
		uint32_t linear<T>::getOutputSize() const
		{
			return this->outputSize;
		}

		template<typename T>
		tensor<T> linear<T>::getWeights() const
		{
			return this->weights;
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