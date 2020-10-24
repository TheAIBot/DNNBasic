#pragma once

#include "optimizer.h"
#include <vector>

namespace dnnbasic
{
	namespace activations
	{
		template<typename T>
		class activationFunction;
	}

	template<typename T>
	class tensor;

	template<typename T>
	class fbpropagation
	{
	public:
		virtual tensor<T> forward(const tensor<T>& x) const = 0;
		virtual void backward(const tensor<T>& estimatedLoss, optimizer::optimizer* opti, std::vector<activations::activationFunction<T>*> actFuncs, bool isFirstLayer) const = 0;
		virtual void updateWeight(const tensor<T>& gradient) 
		{

		}
		tensor<T> operator()(const tensor<T>& x)
		{
			return forward(x);
		}

	private:

	};

}