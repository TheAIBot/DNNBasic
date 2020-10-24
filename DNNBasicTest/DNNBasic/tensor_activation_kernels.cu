#include <cuda_runtime.h>
#include "tensor_activation_kernels.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

namespace dnnbasic
{
	static const uint32_t THREADS_PER_BLOCK = 256;

	template <typename T>
	__device__ T max(const T a, const T b)
	{
		return a > b ? a : b;
	}

	template <typename T>
	__global__ void ReLUKernel(const cudabasic::span<T> input, cudabasic::span<T> output )
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= input.size())
		{
			return;
		}
		output[idx] = max(input[idx], (T)0);
	}

	template <typename T>
	__global__ void ReLUKernelDerivative(const cudabasic::span<T> input, cudabasic::span<T> output)
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= input.size()) 
		{
			return;
		}
		output[idx] = input[idx] > 0 ? 1 : 0;
	}

	// make derivative for backward
	template<typename T>
	void tensorReLUDerivative(const tensor<T>& input, tensor<T>& output)
	{
		const dim3 blockDim(THREADS_PER_BLOCK);
		const dim3 gridDim(integerCeilDivision(input.elementCount(), blockDim.x));

		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addKernelNode(ReLUKernelDerivative<T>, blockDim, gridDim, 0, input.getGPUArrayConst(), output.getGPUArray());
		}
		else
		{
			cudabasic::executeKernel(ReLUKernelDerivative<T>, blockDim, gridDim, 0, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray());
		}
	}

	template<typename T>
	void tensorReLU(const tensor<T>& input, tensor<T>& output)
	{
		const dim3 blockDim(THREADS_PER_BLOCK);
		const dim3 gridDim(integerCeilDivision(input.elementCount(), blockDim.x));

		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addKernelNode(ReLUKernel<T>, blockDim, gridDim, 0, input.getGPUArrayConst(), output.getGPUArray());
		}
		else
		{
			cudabasic::executeKernel(ReLUKernel<T>, blockDim, gridDim, 0, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray());
		}
	}

	template void tensorReLUDerivative(const tensor<uint8_t>& input, tensor<uint8_t>& output);
	template void tensorReLUDerivative(const tensor<uint16_t>& input, tensor<uint16_t>& output);
	template void tensorReLUDerivative(const tensor<uint32_t>& input, tensor<uint32_t>& output);
	template void tensorReLUDerivative(const tensor<uint64_t>& input, tensor<uint64_t>& output);
	template void tensorReLUDerivative(const tensor<int8_t>& input, tensor<int8_t>& output);
	template void tensorReLUDerivative(const tensor<int16_t>& input, tensor<int16_t>& output);
	template void tensorReLUDerivative(const tensor<int32_t>& input, tensor<int32_t>& output);
	template void tensorReLUDerivative(const tensor<int64_t>& input, tensor<int64_t>& output);
	template void tensorReLUDerivative(const tensor<float>& input, tensor<float>& output);
	template void tensorReLUDerivative(const tensor<double>& input, tensor<double>& output);

	template void tensorReLU(const tensor<uint8_t>& input, tensor<uint8_t>& output);
	template void tensorReLU(const tensor<uint16_t>& input, tensor<uint16_t>& output);
	template void tensorReLU(const tensor<uint32_t>& input, tensor<uint32_t>& output);
	template void tensorReLU(const tensor<uint64_t>& input, tensor<uint64_t>& output);
	template void tensorReLU(const tensor<int8_t>& input, tensor<int8_t>& output);
	template void tensorReLU(const tensor<int16_t>& input, tensor<int16_t>& output);
	template void tensorReLU(const tensor<int32_t>& input, tensor<int32_t>& output);
	template void tensorReLU(const tensor<int64_t>& input, tensor<int64_t>& output);
	template void tensorReLU(const tensor<float>& input, tensor<float>& output);
	template void tensorReLU(const tensor<double>& input, tensor<double>& output);
}