#include <cuda_runtime.h>
#include "tensor_exp_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

namespace dnnbasic
{
	static const uint32_t THREADS_PER_BLOCK = 256;


	template <typename T>
	__global__ void ExpKernel(const cudabasic::span<T> input, cudabasic::span<T> output)
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= input.size())
		{
			return;
		}
		output[idx] = (T)__expf((float)input[idx]);
	}

	template<typename T>
	void tensorExp(const tensor<T>& input, tensor<T>& output)
	{
		const dim3 blockDim(THREADS_PER_BLOCK);
		const dim3 gridDim(integerCeilDivision(input.elementCount(), blockDim.x));

		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addKernelNode(ExpKernel<T>, blockDim, gridDim, 0, input.getGPUArrayConst(), output.getGPUArray());
		}
		else
		{
			cudabasic::executeKernel(ExpKernel<T>, blockDim, gridDim, 0, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray());
		}
	}

	template void tensorExp(const tensor<bool>& input, tensor<bool>& output);
	template void tensorExp(const tensor<uint8_t>& input, tensor<uint8_t>& output);
	template void tensorExp(const tensor<uint16_t>& input, tensor<uint16_t>& output);
	template void tensorExp(const tensor<uint32_t>& input, tensor<uint32_t>& output);
	template void tensorExp(const tensor<uint64_t>& input, tensor<uint64_t>& output);
	template void tensorExp(const tensor<int8_t>& input, tensor<int8_t>& output);
	template void tensorExp(const tensor<int16_t>& input, tensor<int16_t>& output);
	template void tensorExp(const tensor<int32_t>& input, tensor<int32_t>& output);
	template void tensorExp(const tensor<int64_t>& input, tensor<int64_t>& output);
	template void tensorExp(const tensor<float>& input, tensor<float>& output);
	template void tensorExp(const tensor<double>& input, tensor<double>& output);
}