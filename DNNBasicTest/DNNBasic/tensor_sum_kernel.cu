#include <cuda_runtime.h>
#include "tensor_permute_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

namespace dnnbasic
{
	using gpuArray = smallGPUArray<uint32_t, tensor<uint32_t>::MAX_DIMENSION_COUNT>;

	static const uint32_t THREADS_PER_BLOCK = 1024;
	static const uint32_t THREADS_PER_WARP = 32;

	template<typename T>
	__global__ void sumKernel(
		const cudabasic::span<T> input, 
		cudabasic::span<T> output,
		const uint32_t sumStride,
		const uint32_t sumDimSize,
		const gpuArray inputStrides,
		const gpuArray outputStrides)
	{
		extern __shared__ __align__(sizeof(T)) int8_t sharedArray[];
		T* sharedMemT = reinterpret_cast<T*>(sharedArray);

		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx >= output.size())
		{
			return;
		}

		uint32_t index[gpuArray::MAX_LENGTH];
		uint32_t x = idx;
		for (uint32_t i = 0; i < outputStrides.size(); i++)
		{
			index[i] = x / outputStrides[i];
			x = x % outputStrides[i];
		}

		uint32_t inputIndex = 0;
		for (size_t i = 0; i < outputStrides.size(); i++)
		{
			inputIndex += index[i] * inputStrides[i];
		}

		const T value = input[inputIndex + threadIdx.x * sumStride];

		//Make warp sum
		T warpSum = value;
		for (uint32_t i = THREADS_PER_WARP / 2; i > 0; i /= 2)
		{
			warpSum += __shfl_down_sync(0xffffffff, warpSum, i);
		}

		//First thread in each warp will store their sum
		//in shared memory so the first warp can sum it up
		if (threadIdx.x % THREADS_PER_WARP == 0)
		{
			sharedMemT[threadIdx.x / THREADS_PER_WARP] = warpSum;
		}
		__syncthreads();

		//First warp in each block will now
		//make a block sum
		T blockSum = 0;
		if (threadIdx.x < THREADS_PER_WARP)
		{
			blockSum = sharedMemT[threadIdx.x];
			for (uint32_t i = THREADS_PER_WARP / 2; i > 0; i /= 2)
			{
				blockSum += __shfl_down_sync(0xffffffff, blockSum, i);
			}
		}
		__syncthreads();

		//First thread in block will now atomic add the result
		
		
		
		for (uint32_t i = 0; i < sumDimSize; i++)
		{
			sum += input[inputIndex + i * sumStride];
		}

		output[idx] = sum;
	}

	template<typename T>
	void tensorSum(const tensor<T>& input, tensor<T>& output, const uint32_t sumDimIdx)
	{
		gpuArray inputStrides(input.getDimensions().size());
		gpuArray outputStrides(input.getDimensions().size());

		for (uint32_t i = 0; i < input.getDimensions().size(); i++)
		{
			uint32_t stride = 1;

			// To get the correct stride when using an array we multiply the following dimensions
			// together such that they correspond to accessing index i of the corresponding matrix 
			// with similar dimensions
			for (uint32_t g = i + 1; g < input.getDimensions().size(); g++)
			{
				stride *= input.getDimensions()[g].dim;
			}
			// if dimension is broadcasted then the stride should be 0 to reuse the same matrix again
			inputStrides[i] = stride;
			outputStrides[i] = stride;
		}

		inputStrides[sumDimIdx] = 0;

		uint32_t sumStride = 1;
		for (size_t i = sumDimIdx + 1; i < input.getDimensions().size(); i++)
		{
			sumStride *= input.getDimensions()[i].dim;
		}

		const uint32_t sumDim = input.getDimensions()[sumDimIdx].dim;
		const uint32_t threadsPerDim = sumDim;
		const uint32_t dimsToSum = output.elementCount();
		const uint32_t totalThreadsNeeded = threadsPerDim * dimsToSum;

		const dim3 blockDim(1024);
		const dim3 gridDim(integerCeilDivision(totalThreadsNeeded, blockDim.x));
		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addKernelNode(sumKernel<T>, blockDim, gridDim, 0, input.getGPUArrayConst(), output.getGPUArray(), sumStride, input.getDimensions()[sumDimIdx].dim, inputStrides, outputStrides);
		}
		else
		{
			cudabasic::executeKernel(sumKernel<T>, blockDim, gridDim, 0, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray(), sumStride, input.getDimensions()[sumDimIdx].dim, inputStrides, outputStrides);
		}
	}

	template void tensorSum(const tensor<bool>& input, tensor<bool>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<uint8_t>& input, tensor<uint8_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<uint16_t>& input, tensor<uint16_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<uint32_t>& input, tensor<uint32_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<uint64_t>& input, tensor<uint64_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<int8_t>& input, tensor<int8_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<int16_t>& input, tensor<int16_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<int32_t>& input, tensor<int32_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<int64_t>& input, tensor<int64_t>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<float>& input, tensor<float>& output, const uint32_t sumDimIdx);
	template void tensorSum(const tensor<double>& input, tensor<double>& output, const uint32_t sumDimIdx);
}