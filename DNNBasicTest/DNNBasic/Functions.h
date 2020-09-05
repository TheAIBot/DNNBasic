#pragma once
#include "FBPropagation.h"
#include "Tensor.h"

namespace dnnbasic::activation
{
	template<typename T>
	class sigmoid : fbpropagation<T>
	{
	public:
		tensor<T>& forward(const tensor<T>& x) const override
		{
		}
		tensor<T>& backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{

		}
	private:

	};

	template<typename T>
	class tanh : fbpropagation<T>
	{
	public:
		tensor<T>& forward(const tensor<T>& x) const override
		{
		}
		tensor<T>& backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{

		}

	private:

	};

	template<typename T>
	class swish : fbpropagation<T>
	{
	public:
		tensor<T>& forward(const tensor<T>& x) const override
		{
		}
		tensor<T>& backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{

		}

	private:

	};

	template<typename T>
	class softmax : fbpropagation<T>
	{
	public:
		tensor<T>& forward(const tensor<T>& x) const override
		{
		}
		tensor<T>& backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{

		}

	private:

	};

	template<typename T>
	class ReLU : fbpropagation<T>
	{
	private:

	public:
		tensor<T>& forward(const tensor<T>& x) const override
		{
		}
		tensor<T>& backward(const tensor<T>& estimatedLoss, const tensor<T>& functionOut) const override
		{

		}
	};

}
