#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include "span.h"
#include "tensor_def.h"

namespace dnnbasic
{
	template<typename T>
	__global__ void multiplyGPU(cudabasic::span<T> left, cudabasic::span<T> right, cudabasic::span<T> output);

	template<typename T>
	void tensorMultiply(tensor<T>& left, tensor<T>& right, tensor<T>& result);

	template<>
	__global__ void multiplyGPU<float>(cudabasic::span<float> left, cudabasic::span<float> right, cudabasic::span<float> output);

	template<>
	void tensorMultiply<float>(tensor<float>& left, tensor<float>& right, tensor<float>& result);
}