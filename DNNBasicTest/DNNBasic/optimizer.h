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
			virtual void updateWeights(tensor<bool>& weights, const tensor<bool>& gradients) = 0;
			virtual void updateWeights(tensor<uint8_t>& weights, const tensor<uint8_t>& gradients) = 0;
			virtual void updateWeights(tensor<uint16_t>& weights, const tensor<uint16_t>& gradients) = 0;
			virtual void updateWeights(tensor<uint32_t>& weights, const tensor<uint32_t>& gradients) = 0;
			virtual void updateWeights(tensor<uint64_t>& weights, const tensor<uint64_t>& gradients) = 0;
			virtual void updateWeights(tensor<int8_t>& weights, const tensor<int8_t>& gradients) = 0;
			virtual void updateWeights(tensor<int16_t>& weights, const tensor<int16_t>& gradients) = 0;
			virtual void updateWeights(tensor<int32_t>& weights, const tensor<int32_t>& gradients) = 0;
			virtual void updateWeights(tensor<int64_t>& weights, const tensor<int64_t>& gradients) = 0;
			virtual void updateWeights(tensor<float>& weights, const tensor<float>& gradients) = 0;
			virtual void updateWeights(tensor<double>& weights, const tensor<double>& gradients) = 0;
		};
	}
}