#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "tensor_def.h"

namespace dnnbasic 
{
	void tensorMatrixMul(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result);
	void tensorMatrixMul(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result);
	void tensorMatrixMul(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result);
	void tensorMatrixMul(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result);
	void tensorMatrixMul(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result);
	void tensorMatrixMul(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result);
	void tensorMatrixMul(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result);
	void tensorMatrixMul(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result);
	void tensorMatrixMul(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result);
	void tensorMatrixMul(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
	void tensorMatrixMul(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result);

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