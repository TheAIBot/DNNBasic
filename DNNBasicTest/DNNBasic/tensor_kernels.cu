#include "tensor_kernels.cuh"

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
	void multiply<float>(tensor<float>& left, tensor<float>& right, tensor<float>& result)
	{
		dim3 blockDim(256);
		dim3 gridDim((left.elementCount() + (blockDim.x - 1)) / blockDim.x);

		cudabasic::cpuGpuArray<float> inputA(left.elementCount());
		cudabasic::cpuGpuArray<float> inputB(right.elementCount());
		cudabasic::cpuGpuArray<float> output(left.elementCount());

		std::copy(left.arr.begin(), left.arr.end(), inputA.getCPUArray().begin());
		std::copy(right.arr.begin(), right.arr.end(), inputB.getCPUArray().begin());

		inputA.copyToGPU();
		inputB.copyToGPU();

		cudabasic::executeKernel(multiplyGPU, blockDim, gridDim, inputA.getGPUArray(), inputB.getGPUArray(), output.getGPUArray());

		cudabasic::span<float> mulResult = output.copyFromGPU();
		std::copy(mulResult.begin(), mulResult.end(), result.arr.begin());
	}
}