#include <cuda_runtime.h>
#include "tensor_log_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

namespace dnnbasic
{
	static const uint32_t THREADS_PER_BLOCK = 256;


	template <typename T>
	__global__ void LogKernel(const cudabasic::span<T> input, cudabasic::span<T> output)
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= input.size())
		{
			return;
		}
		output[idx] = (T)__logf((float)input[idx]);
	}

	template<typename T>
	void tensorLog(const tensor<T>& input, tensor<T>& output)
	{
		const dim3 blockDim(THREADS_PER_BLOCK);
		const dim3 gridDim(integerCeilDivision(input.elementCount(), blockDim.x));

		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addKernelNode(LogKernel<T>, blockDim, gridDim, 0, input.getGPUArrayConst(), output.getGPUArray());
		}
		else
		{
			cudabasic::executeKernel(LogKernel<T>, blockDim, gridDim, 0, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray());
		}
	}

	template void tensorLog(const tensor<bool>& input, tensor<bool>& output);
	template void tensorLog(const tensor<uint8_t>& input, tensor<uint8_t>& output);
	template void tensorLog(const tensor<uint16_t>& input, tensor<uint16_t>& output);
	template void tensorLog(const tensor<uint32_t>& input, tensor<uint32_t>& output);
	template void tensorLog(const tensor<uint64_t>& input, tensor<uint64_t>& output);
	template void tensorLog(const tensor<int8_t>& input, tensor<int8_t>& output);
	template void tensorLog(const tensor<int16_t>& input, tensor<int16_t>& output);
	template void tensorLog(const tensor<int32_t>& input, tensor<int32_t>& output);
	template void tensorLog(const tensor<int64_t>& input, tensor<int64_t>& output);
	template void tensorLog(const tensor<float>& input, tensor<float>& output);
	template void tensorLog(const tensor<double>& input, tensor<double>& output);
}