#pragma once

#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	void tensorMultiDimMatrixMul(const tensor<bool>& a, const tensor<bool>& b, const tensor<bool>& c);
	void tensorMultiDimMatrixMul(const tensor<uint8_t>& a, const tensor<uint8_t>& b, tensor<uint8_t>& c);
	void tensorMultiDimMatrixMul(const tensor<uint16_t>& a, const tensor<uint16_t>& b, tensor<uint16_t>& c);
	void tensorMultiDimMatrixMul(const tensor<uint32_t>& a, const tensor<uint32_t>& b, tensor<uint32_t>& c);
	void tensorMultiDimMatrixMul(const tensor<uint64_t>& a, const tensor<uint64_t>& b, tensor<uint64_t>& c);
	void tensorMultiDimMatrixMul(const tensor<int8_t>& a, const tensor<int8_t>& b, tensor<int8_t>& c);
	void tensorMultiDimMatrixMul(const tensor<int16_t>& a, const tensor<int16_t>& b, tensor<int16_t>& c);
	void tensorMultiDimMatrixMul(const tensor<int32_t>& a, const tensor<int32_t>& b, tensor<int32_t>& c);
	void tensorMultiDimMatrixMul(const tensor<int64_t>& a, const tensor<int64_t>& b, tensor<int64_t>& c);
	void tensorMultiDimMatrixMul(const tensor<float>& a, const tensor<float>& b, tensor<float>& c);
	void tensorMultiDimMatrixMul(const tensor<double>& a, const tensor<double>& b, tensor<double>& c);
}