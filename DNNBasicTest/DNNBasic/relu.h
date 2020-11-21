#pragma once
#include "FBPropagation.h"
#include "tensor.h"
#include "activation_function.h"

namespace dnnbasic
{
	namespace activations
	{
		template<typename T>
		class relu :public activationFunction<T>
		{
		public:
			tensor<T> forward(const tensor<T>& x) override;
			tensor<T> derivative(const tensor<T>& derivative_activation_function, const tensor<T>& affine_input) override;
		};
	}
}
