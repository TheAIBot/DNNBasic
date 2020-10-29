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
			weights(tensor<T>::random({ inputDim, outputDim })),
			biases(tensor<T>::random({ useBias ? outputDim : 1 })),
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
			tensor<T> newDerivative = actFuncs.size() == 0 ? tensor<T>(std::vector<uint32_t>({ 1 }), std::vector<T>({ 1 })) : actFuncs.back()->derivative(output);
			if (actFuncs.size()>1)
			{
				tensor<T> forwardDerivative = actFuncs.back()->forward(output);
				for (int i = actFuncs.size() - 2; i >= 0; i--)
				{
					newDerivative = newDerivative * actFuncs[i]->derivative(forwardDerivative);
					forwardDerivative = actFuncs[i]->forward(forwardDerivative);
				}
			}

			// error for layer L
			const tensor<T> newLoss = estimatedLoss * newDerivative;

			const tensor<T> awd = input.transpose(input.getDimensions().size() - 1, input.getDimensions().size() - 2);
			const tensor<T> inputOuterShape = awd.reshape(awd.getDimension(0), awd.getDimension(1), 1);
			const tensor<T> newLossOuterShape = newLoss.reshape(newLoss.getDimension(0), 1, newLoss.getDimension(1));

			const tensor<T> batchWeightGradient = inputOuterShape.matMul(newLossOuterShape);
			const tensor<T> meanWeightGradient = batchWeightGradient.sum(0) / ((T)batchWeightGradient.getDimension(0));

			// Partial derivative cost for weight
			opti->updateWeights(this->weights, meanWeightGradient);

			// Partial derivative cost for bias
			if (this->useBias)
			{
				opti->updateWeights(this->biases, newLoss.sum(0)/((T)newLoss.getDimension(0)));
			}

			return newLoss.matMul(this->weights.permute({ 1, 0 }));
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