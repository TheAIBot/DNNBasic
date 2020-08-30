#pragma once
#include "Tensor.h"
namespace dnnbasic
{
	template<typename T>
	class fbpropagation
	{
	public:
		virtual tensor<T>& forward(const tensor<T>& x) const = 0;
		virtual tensor<T>& backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const = 0;
		virtual void updateWeight(const tensor<T>& gradient) 
		{

		}
		tensor<T>& operator(const tensor<T>& x)
		{
			return forward(x);
		}

	private:

	};

}