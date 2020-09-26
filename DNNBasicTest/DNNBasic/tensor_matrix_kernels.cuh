#pragma once

#include <cstdint>
#include "matrix.h"
#include <cuda_runtime.h>

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
}