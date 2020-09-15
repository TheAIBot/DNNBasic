#pragma once

#include <vector>
#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	void tensorPermute(const tensor<bool>& input, const tensor<bool>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<uint8_t>& input, const tensor<uint8_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<uint16_t>& input, const tensor<uint16_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<uint32_t>& input, const tensor<uint32_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<uint64_t>& input, const tensor<uint64_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<int8_t>& input, const tensor<int8_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<int16_t>& input, const tensor<int16_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<int32_t>& input, const tensor<int32_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<int64_t>& input, const tensor<int64_t>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<float>& input, const tensor<float>& output, const std::vector<uint32_t>& dims);
	void tensorPermute(const tensor<double>& input, const tensor<double>& output, const std::vector<uint32_t>& dims);
}