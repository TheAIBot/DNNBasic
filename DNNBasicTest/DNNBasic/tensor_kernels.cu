#include "tensor_kernels.cuh"
#include "cudaBasics.h"

namespace dnnbasic
{
	template<typename T>
	__global__ void multiplyGPU(cudabasic::span<T> left, cudabasic::span<T> right, cudabasic::span<T> output)
	{
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= left.size())
		{
			return;
		}
		output[index] = left[index] * right[index];
	}

	void tensorMultiply(tensor<float>& left, tensor<float>& right, tensor<float>& result)
	{
		dim3 blockDim(256);
		dim3 gridDim((left.elementCount() + (blockDim.x - 1)) / blockDim.x);

		cudabasic::executeKernel(multiplyGPU, blockDim, gridDim, left.getGPUArray(), right.getGPUArray(), result.getGPUArray());
	}
}