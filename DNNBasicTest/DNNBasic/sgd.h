#pragma once

#include <cstdint>
#include "optimizer.h"

namespace dnnbasic::optimizer
{
	class sgd : public optimizer
	{
	private:
		float learningRate;

		template<typename T>
		void update(tensor<T>& weights, const tensor<T>& gradients);

	public:
		sgd(float learningRate);

		void updateWeights(tensor<bool>& weights, const tensor<bool>& gradients);
		void updateWeights(tensor<uint8_t>& weights, const tensor<uint8_t>& gradients);
		void updateWeights(tensor<uint16_t>& weights, const tensor<uint16_t>& gradients);
		void updateWeights(tensor<uint32_t>& weights, const tensor<uint32_t>& gradients);
		void updateWeights(tensor<uint64_t>& weights, const tensor<uint64_t>& gradients);
		void updateWeights(tensor<int8_t>& weights, const tensor<int8_t>& gradients);
		void updateWeights(tensor<int16_t>& weights, const tensor<int16_t>& gradients);
		void updateWeights(tensor<int32_t>& weights, const tensor<int32_t>& gradients);
		void updateWeights(tensor<int64_t>& weights, const tensor<int64_t>& gradients);
		void updateWeights(tensor<float>& weights, const tensor<float>& gradients);
		void updateWeights(tensor<double>& weights, const tensor<double>& gradients);
	};
}