#pragma once
#include "FBPropagation.h"
#include "tensor.h"

namespace dnnbasic
{
	namespace activations
	{
		template<typename T>
		class activationFunction
		{
		public:
			virtual tensor<T> forward(const tensor<T>& input)=0;
			virtual tensor<T> derivative(const tensor<T>& input)=0;
		};
	}
}
