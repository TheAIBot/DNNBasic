#pragma once
#include "FBPropagation.h"
#include "tensor.h"

namespace dnnbasic
{
	namespace layer
	{
		template<typename T>
		class linear
		{
		private:
			tensor<T> weights;
			tensor<T> biases;
			bool useBias;
			uint32_t inputSize;
			uint32_t outputSize;

		public:
			linear(const uint32_t inputDim, const uint32_t outputDim, const bool useBias);

			tensor<T> forward(const tensor<T>& x);
			tensor<T> backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti, const tensor<T>& input, const tensor<T>& output, std::vector<activations::activationFunction<T>*> actFuncs, bool isFirstLayer);

			uint32_t getInputSize() const;
			uint32_t getOutputSize() const;
			tensor<T> getWeights() const;
		};
	}
}
