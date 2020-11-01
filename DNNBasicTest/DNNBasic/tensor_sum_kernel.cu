#include <cuda_runtime.h>
#include <type_traits>
#include "tensor_permute_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

__device__ void __syncthreads();

namespace dnnbasic
{
	using gpuArray = smallGPUArray<uint32_t, tensor<uint32_t>::MAX_DIMENSION_COUNT>;

	static const uint32_t THREADS_PER_BLOCK = 1024;
	static const uint32_t THREADS_PER_WARP = 32;
	static const uint32_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

	template<typename T>
	__device__ T getWarpSum(const T threadValue)
	{
		T warpSum = threadValue;
		for (uint32_t i = THREADS_PER_WARP / 2; i > 0; i /= 2)
		{
			warpSum += __shfl_down_sync(0xffffffff, warpSum, i);
		}

		return warpSum;
	}

	template<typename T>
	__global__ void sumKernel(
		const cudabasic::span<T> input, 
		cudabasic::span<T> output,
		const uint32_t sumElementStride,
		const uint32_t sumDimSize,
		const uint32_t sumsToMake,
		const uint32_t blocksMade)
	{
		extern __shared__ __align__(sizeof(T)) int8_t sharedArray[];
		T* sharedMemT = reinterpret_cast<T*>(sharedArray);

		const uint32_t sumElemIdx = blockIdx.x * blockDim.x + threadIdx.x;

		for (uint32_t i = blockIdx.y; i < sumsToMake; i += blocksMade)
		{
			//if index is out of bounds then load zero instead
			//as all threads in a warp are needed to sum
			//and keeping all threads to begin with it the
			//easiest way to do that
			const T value = sumElemIdx >= sumDimSize ? 0 : input[sumElemIdx * sumElementStride + (i / sumElementStride) * sumElementStride * sumDimSize + (i % sumElementStride)];

			//Make warp sum
			const T warpSum = getWarpSum(value);

			//First thread in each warp will store their sum
			//in shared memory so the first warp can sum it up
			if (threadIdx.x % THREADS_PER_WARP == 0)
			{
				sharedMemT[threadIdx.x / WARPS_PER_BLOCK] = warpSum;
			}
			__syncthreads();

			//First warp in each block will now
			//make a block sum
			T blockSum = 0;
			if (threadIdx.x < WARPS_PER_BLOCK)
			{
				blockSum = getWarpSum(sharedMemT[threadIdx.x]);
			}
			__syncthreads();

			//First thread in block will now atomic add the result
			if (threadIdx.x == 0)
			{
				if constexpr (std::is_integral<T>::value && std::is_signed<T>::value)
				{
					using unsigned_T = typename std::make_unsigned<T>::type;
					atomicAdd(reinterpret_cast<unsigned_T*>(&output[blockIdx.y]), (unsigned_T)blockSum);
				}
				else
				{
					atomicAdd(&output[blockIdx.y], blockSum);
				}
			}
		}
	}

	template<typename T>
	void tensorSum(const tensor<T>& input, tensor<T>& output, const uint32_t sumDimIdx)
	{
		if constexpr (sizeof(T) < 4)
		{
			throw std::runtime_error("Sum is currently not supported for that tensor type.");
		}
		else
		{
			uint32_t sumElementStride = 1;
			for (size_t i = sumDimIdx + 1; i < input.getDimensions().size(); i++)
			{
				sumElementStride *= input.getDimensions()[i].dim;
			}

			const uint32_t sumDim = input.getDimensions()[sumDimIdx].dim;
			const uint32_t dimsToSum = output.elementCount();
			const uint32_t blocksMade = std::min(dimsToSum, 40u);

			const dim3 blockDim(THREADS_PER_BLOCK);
			const dim3 gridDim(integerCeilDivision(sumDim, blockDim.x), blocksMade);
			if (autoGraph::isRecordingGraph())
			{
				const std::vector<void*> inputPtrs = { reinterpret_cast<void*>(input.getGPUArray().begin()), reinterpret_cast<void*>(output.getGPUArray().begin()) };
				const void* outputPtr = reinterpret_cast<void*>(output.getGPUArray().begin());
				autoGraph::addMemsetNode(outputPtr, outputPtr, output.getGPUArray(), 0);
				autoGraph::addKernelNode(inputPtrs, outputPtr, sumKernel<T>, blockDim, gridDim, (uint32_t)sizeof(T) * WARPS_PER_BLOCK, input.getGPUArrayConst(), output.getGPUArray(), sumElementStride, sumDim, dimsToSum, blocksMade);
			}
			else
			{
				cudaMemset(output.getGPUArray().begin(), 0, output.elementCount() * sizeof(T));
				cudabasic::executeKernel(sumKernel<T>, blockDim, gridDim, sizeof(T) * WARPS_PER_BLOCK, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray(), sumElementStride, sumDim, dimsToSum, blocksMade);
			}
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