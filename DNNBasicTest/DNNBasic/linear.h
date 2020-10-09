#pragma once
#include "FBPropagation.h"
#include "tensor.h"

namespace dnnbasic
{
	namespace layer
	{
		template<typename T>
		class linear : fbpropagation<T>
		{
		private:
			tensor<T> weights;
			tensor<T> biases;
			bool useBias;

		public:
			linear(const uint32_t inputDim, const uint32_t outputDim, const bool useBias);

			tensor<T> forward(const tensor<T>& x) const override;
			void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti) const override;
		};
	}
}
