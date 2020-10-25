#pragma once

#include <cstdint>
#include <vector>
#include <tuple>
#include "tensor.h"

namespace dnnbasic::datasets
{
	template<typename T>
	class supervisedDataset
	{
	private:
		std::vector<tensor<uint8_t>> inputs;
		std::vector<tensor<uint8_t>> outputs;

	public:
		supervisedDataset(std::vector<tensor<uint8_t>>& inputs, std::vector<tensor<uint8_t>>& outputs) : inputs(inputs), outputs(outputs)
		{ }

		std::tuple<tensor<uint8_t>, tensor<uint8_t>> operator[](const size_t index) const
		{
			return std::make_tuple(inputs[index], outputs[index]);
		}

		size_t getBatchCount() const
		{
			return inputs.size();
		}
	};
}