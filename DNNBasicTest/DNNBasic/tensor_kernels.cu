#include "tensor_kernels.cuh"
#include "cudaBasics.h"

namespace dnnbasic
{
	template<typename T>
	__global__ void multiplyGPU(const cudabasic::span<T> left, const cudabasic::span<T> right, cudabasic::span<T> output)
	{
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= left.size())
		{
			return;
		}
		output[index] = left[index] * right[index];
	}

	template<typename T>
	__global__ void addGPU(const cudabasic::span<T> left, const cudabasic::span<T> right, cudabasic::span<T> output)
	{
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= output.size())
		{
			return;
		}
		output[index] = left[index] + right[index];
	}

	template<typename T>
	__global__ void addGPU(const T left, const cudabasic::span<T> right, cudabasic::span<T> output)
	{
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= output.size())
		{
			return;
		}
		output[index] = left + right[index];
	}

	void tensorMultiply(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result)
	{
		const dim3 blockDim(256);
		const dim3 gridDim((left.elementCount() + (blockDim.x - 1)) / blockDim.x);

		cudabasic::executeKernel(multiplyGPU, blockDim, gridDim, left.getGPUArray(), right.getGPUArray(), result.getGPUArray());
	}

	void tensorAdd(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result)
	{
		const dim3 blockDim(256);
		const dim3 gridDim((result.elementCount() + (blockDim.x - 1)) / blockDim.x);

		cudabasic::executeKernel(addGPU, blockDim, gridDim, left.getGPUArray(), right.getGPUArray(), result.getGPUArray());
	}

	void tensorAdd(const float left, const tensor<float>& right, const tensor<float>& result)
	{
		const dim3 blockDim(256);
		const dim3 gridDim((result.elementCount() + (blockDim.x - 1)) / blockDim.x);

		cudabasic::executeKernel(addGPU, blockDim, gridDim, left, right.getGPUArray(), result.getGPUArray());
	}
}