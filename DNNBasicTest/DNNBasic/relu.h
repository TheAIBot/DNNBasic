#pragma once
#include "FBPropagation.h"
#include "tensor.h"
#include "activation.h"

namespace dnnbasic
{
	namespace activations
	{
		template<typename T>
		class relu :public activationFunction<T>
		{
		public:
			tensor<T> forward(const tensor<T>& x) override;
			tensor<T> derivative(const tensor<T>& input) override;
		};
	}
}
