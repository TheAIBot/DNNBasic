#include <cuda_runtime.h>
#include <vector>
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
	__global__ void ReLUKernelDerivative(const cudabasic::span<T> derivative_activation_function, const cudabasic::span<T> affine_input, cudabasic::span<T> output)
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= derivative_activation_function.size())
		{
			return;
		}
		output[idx] = affine_input[idx] > 0 ? derivative_activation_function[idx] : 0;
	}

	// make derivative for backward
	template<typename T>
	void tensorReLUDerivative(const tensor<T>& derivative_activation_function, const tensor<T>& affine_input, tensor<T>& output)
	{
		const dim3 blockDim(THREADS_PER_BLOCK);
		const dim3 gridDim(integerCeilDivision(affine_input.elementCount(), blockDim.x));

		if (autoGraph::isRecordingGraph())
		{
			const std::vector<void*> inputsPtrs = { reinterpret_cast<void*>(derivative_activation_function.getGPUArray().begin()), reinterpret_cast<void*>(affine_input.getGPUArray().begin()) };
			const void* outputPtr = reinterpret_cast<void*>(output.getGPUArray().begin());
			autoGraph::addKernelNode(inputsPtrs, outputPtr, ReLUKernelDerivative<T>, blockDim, gridDim, 0, derivative_activation_function.getGPUArrayConst(), affine_input.getGPUArrayConst(), output.getGPUArray());
		}
		else
		{
			cudabasic::executeKernel(ReLUKernelDerivative<T>, blockDim, gridDim, 0, cuda::getDefaultStream(), derivative_activation_function.getGPUArrayConst(), affine_input.getGPUArrayConst(), output.getGPUArray());
		}
	}

	template<typename T>
	void tensorReLU(const tensor<T>& input, tensor<T>& output)
	{
		const dim3 blockDim(THREADS_PER_BLOCK);
		const dim3 gridDim(integerCeilDivision(input.elementCount(), blockDim.x));

		if (autoGraph::isRecordingGraph())
		{
			const std::vector<void*> inputsPtrs = { reinterpret_cast<void*>(input.getGPUArray().begin()) };
			const void* outputPtr = reinterpret_cast<void*>(output.getGPUArray().begin());
			autoGraph::addKernelNode(inputsPtrs, outputPtr, ReLUKernel<T>, blockDim, gridDim, 0, input.getGPUArrayConst(), output.getGPUArray());
		}
		else
		{
			cudabasic::executeKernel(ReLUKernel<T>, blockDim, gridDim, 0, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray());
		}
	}

	template void tensorReLUDerivative(const tensor<uint8_t>& derivative_activation_function, const tensor<uint8_t>& affine_input, tensor<uint8_t>& output);
	template void tensorReLUDerivative(const tensor<uint16_t>& derivative_activation_function, const tensor<uint16_t>& affine_input, tensor<uint16_t>& output);
	template void tensorReLUDerivative(const tensor<uint32_t>& derivative_activation_function, const tensor<uint32_t>& affine_input, tensor<uint32_t>& output);
	template void tensorReLUDerivative(const tensor<uint64_t>& derivative_activation_function, const tensor<uint64_t>& affine_input, tensor<uint64_t>& output);
	template void tensorReLUDerivative(const tensor<int8_t>& derivative_activation_function, const tensor<int8_t>& affine_input, tensor<int8_t>& output);
	template void tensorReLUDerivative(const tensor<int16_t>& derivative_activation_function, const tensor<int16_t>& affine_input, tensor<int16_t>& output);
	template void tensorReLUDerivative(const tensor<int32_t>& derivative_activation_function, const tensor<int32_t>& affine_input, tensor<int32_t>& output);
	template void tensorReLUDerivative(const tensor<int64_t>& derivative_activation_function, const tensor<int64_t>& affine_input, tensor<int64_t>& output);
	template void tensorReLUDerivative(const tensor<float>& derivative_activation_function, const tensor<float>& affine_input, tensor<float>& output);
	template void tensorReLUDerivative(const tensor<double>& derivative_activation_function, const tensor<double>& affine_input, tensor<double>& output);

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