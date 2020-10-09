#pragma once

#include <stdexcept>
#include "optimizer.h"
#include "tensor_elementwise_kernels.cuh"

namespace dnnbasic::optimizer
{
	class sgd : optimizer
	{
	private:
		float learningRate;

		template<typename T>
		void update(tensor<T>& weights, tensor<T>& gradients)
		{
			auto lrGrad = (gradients.cast<float>() * this->learningRate).cast<T>();
			tensorSubtract(weights, lrGrad, weights);
		}

	public:
		sgd(float learningRate) : learningRate(learningRate)
		{ }

		void updateWeights(tensor<bool>& weights, tensor<bool>& gradients)
		{
			throw new std::runtime_error("Can't update weights for bool tensors yet.");
		}

		void updateWeights(tensor<uint8_t>& weights, tensor<uint8_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<uint16_t>& weights, tensor<uint16_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<uint32_t>& weights, tensor<uint32_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<uint64_t>& weights, tensor<uint64_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<int8_t>& weights, tensor<int8_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<int16_t>& weights, tensor<int16_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<int32_t>& weights, tensor<int32_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<int64_t>& weights, tensor<int64_t>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<float>& weights, tensor<float>& gradients) { update(weights, gradients); }
		void updateWeights(tensor<double>& weights, tensor<double>& gradients) { update(weights, gradients); }
	};
}