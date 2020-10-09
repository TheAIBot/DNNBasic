#pragma once

#include <cstdint>

namespace dnnbasic
{
	template<typename T>
	class tensor;

	namespace optimizer
	{
		class optimizer
		{
		public:
			virtual void updateWeights(tensor<bool>& weights, tensor<bool>& gradients) = 0;
			virtual void updateWeights(tensor<uint8_t>& weights, tensor<uint8_t>& gradients) = 0;
			virtual void updateWeights(tensor<uint16_t>& weights, tensor<uint16_t>& gradients) = 0;
			virtual void updateWeights(tensor<uint32_t>& weights, tensor<uint32_t>& gradients) = 0;
			virtual void updateWeights(tensor<uint64_t>& weights, tensor<uint64_t>& gradients) = 0;
			virtual void updateWeights(tensor<int8_t>& weights, tensor<int8_t>& gradients) = 0;
			virtual void updateWeights(tensor<int16_t>& weights, tensor<int16_t>& gradients) = 0;
			virtual void updateWeights(tensor<int32_t>& weights, tensor<int32_t>& gradients) = 0;
			virtual void updateWeights(tensor<int64_t>& weights, tensor<int64_t>& gradients) = 0;
			virtual void updateWeights(tensor<float>& weights, tensor<float>& gradients) = 0;
			virtual void updateWeights(tensor<double>& weights, tensor<double>& gradients) = 0;
		};
	}
}