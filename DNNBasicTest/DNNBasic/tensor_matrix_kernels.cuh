#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "tensor.h"

namespace dnnbasic 
{
	void tensorMatrixMul(const matrix<bool>& left, const matrix<bool>& right, matrix<bool>& result);
	void tensorMatrixMul(const matrix<uint8_t>& left, const matrix<uint8_t>& right, matrix<uint8_t>& result);
	void tensorMatrixMul(const matrix<uint16_t>& left, const matrix<uint16_t>& right, matrix<uint16_t>& result);
	void tensorMatrixMul(const matrix<uint32_t>& left, const matrix<uint32_t>& right, matrix<uint32_t>& result);
	void tensorMatrixMul(const matrix<uint64_t>& left, const matrix<uint64_t>& right, matrix<uint64_t>& result);
	void tensorMatrixMul(const matrix<int8_t>& left, const matrix<int8_t>& right, matrix<int8_t>& result);
	void tensorMatrixMul(const matrix<int16_t>& left, const matrix<int16_t>& right, matrix<int16_t>& result);
	void tensorMatrixMul(const matrix<int32_t>& left, const matrix<int32_t>& right, matrix<int32_t>& result);
	void tensorMatrixMul(const matrix<int64_t>& left, const matrix<int64_t>& right, matrix<int64_t>& result);
	void tensorMatrixMul(const matrix<float>& left, const matrix<float>& right, matrix<float>& result);
	void tensorMatrixMul(const matrix<double>& left, const matrix<double>& right, matrix<double>& result);

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