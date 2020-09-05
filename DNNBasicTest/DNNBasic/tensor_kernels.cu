#include "tensor_kernels.cuh"
#include "cudaBasics.h"

namespace dnnbasic
{
	template<>
	__global__ void multiplyGPU<float>(cudabasic::span<float> left, cudabasic::span<float> right, cudabasic::span<float> output)
	{
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= left.size())
		{
			return;
		}
		output[index] = left[index] * right[index];
	}

	template<>
	void tensorMultiply<float>(tensor<float>& left, tensor<float>& right, tensor<float>& result)
	{
		dim3 blockDim(256);
		dim3 gridDim((left.elementCount() + (blockDim.x - 1)) / blockDim.x);

		cudabasic::executeKernel(multiplyGPU, blockDim, gridDim, left.getGPUArray(), right.getGPUArray(), result.getGPUArray());
	}
}