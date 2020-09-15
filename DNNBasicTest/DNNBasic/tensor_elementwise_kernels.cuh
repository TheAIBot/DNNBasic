#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	void tensorMultiply(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result);
	void tensorMultiply(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result);
	void tensorMultiply(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result);
	void tensorMultiply(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result);
	void tensorMultiply(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result);
	void tensorMultiply(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result);
	void tensorMultiply(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result);
	void tensorMultiply(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result);
	void tensorMultiply(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result);
	void tensorMultiply(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
	void tensorMultiply(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result);

	void tensorMultiply(const bool left, const tensor<bool>& right, const tensor<bool>& result);
	void tensorMultiply(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result);
	void tensorMultiply(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result);
	void tensorMultiply(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result);
	void tensorMultiply(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result);
	void tensorMultiply(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result);
	void tensorMultiply(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result);
	void tensorMultiply(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result);
	void tensorMultiply(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result);
	void tensorMultiply(const float left, const tensor<float>& right, const tensor<float>& result);
	void tensorMultiply(const double left, const tensor<double>& right, const tensor<double>& result);

	void tensorAdd(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result);
	void tensorAdd(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result);
	void tensorAdd(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result);
	void tensorAdd(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result);
	void tensorAdd(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result);
	void tensorAdd(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result);
	void tensorAdd(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result);
	void tensorAdd(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result);
	void tensorAdd(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result);
	void tensorAdd(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
	void tensorAdd(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result);

	void tensorAdd(const bool left, const tensor<bool>& right, const tensor<bool>& result);
	void tensorAdd(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result);
	void tensorAdd(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result);
	void tensorAdd(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result);
	void tensorAdd(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result);
	void tensorAdd(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result);
	void tensorAdd(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result);
	void tensorAdd(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result);
	void tensorAdd(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result);
	void tensorAdd(const float left, const tensor<float>& right, const tensor<float>& result);
	void tensorAdd(const double left, const tensor<double>& right, const tensor<double>& result);

	void tensorSubtract(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result);
	void tensorSubtract(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result);
	void tensorSubtract(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result);
	void tensorSubtract(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result);
	void tensorSubtract(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result);
	void tensorSubtract(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result);
	void tensorSubtract(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result);
	void tensorSubtract(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result);
	void tensorSubtract(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result);
	void tensorSubtract(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result);
	void tensorSubtract(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result);

	void tensorSubtract(const bool left, const tensor<bool>& right, const tensor<bool>& result);
	void tensorSubtract(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result);
	void tensorSubtract(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result);
	void tensorSubtract(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result);
	void tensorSubtract(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result);
	void tensorSubtract(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result);
	void tensorSubtract(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result);
	void tensorSubtract(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result);
	void tensorSubtract(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result);
	void tensorSubtract(const float left, const tensor<float>& right, const tensor<float>& result);
	void tensorSubtract(const double left, const tensor<double>& right, const tensor<double>& result);
}