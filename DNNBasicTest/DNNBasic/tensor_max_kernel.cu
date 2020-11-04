#include <cuda_runtime.h>
#include <type_traits>
#include "tensor_max_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

__device__ void __syncthreads();

namespace dnnbasic
{

	template <typename T>
	__device__ T max(const T a, const T b)
	{
		return a > b ? a : b;
	}

	using gpuArray = smallGPUArray<uint32_t, tensor<uint32_t>::MAX_DIMENSION_COUNT>;

	static const uint32_t THREADS_PER_BLOCK = 1024;
	static const uint32_t THREADS_PER_WARP = 32;
	static const uint32_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

	template<typename T>
	__device__ T getWarpMax(const T threadValue)
	{
		T warpSum = threadValue;
		for (uint32_t i = THREADS_PER_WARP / 2; i > 0; i /= 2)
		{
			warpSum = max(warpSum, __shfl_down_sync(0xffffffff, warpSum, i));
		}

		return warpSum;
	}

	template<typename T>
	__global__ void maxKernel(
		const cudabasic::span<T> input,
		cudabasic::span<T> output,
		const uint32_t maxElementStride,
		const uint32_t maxDimSize,
		const uint32_t maxsToMake,
		const uint32_t blocksMade)
	{
		extern __shared__ __align__(sizeof(T)) int8_t sharedArray[];
		T* sharedMemT = reinterpret_cast<T*>(sharedArray);

		const uint32_t maxElemIdx = blockIdx.x * blockDim.x + threadIdx.x;

		for (uint32_t i = blockIdx.y; i < maxsToMake; i += blocksMade)
		{
			//if index is out of bounds then load zero instead
			//as all threads in a warp are needed to sum
			//and keeping all threads to begin with it the
			//easiest way to do that
			const T value = maxElemIdx >= maxDimSize ? 0 : input[maxElemIdx * maxElementStride + (i / maxElementStride) * maxElementStride * maxDimSize + (i % maxElementStride)];

			//Make warp sum
			const T warpMax = getWarpMax(value);

			//First thread in each warp will store their sum
			//in shared memory so the first warp can sum it up
			if (threadIdx.x % THREADS_PER_WARP == 0)
			{
				sharedMemT[threadIdx.x / WARPS_PER_BLOCK] = warpMax;
			}
			__syncthreads();

			//First warp in each block will now
			//make a block sum
			T blockSum = 0;
			if (threadIdx.x < WARPS_PER_BLOCK)
			{
				blockSum = getWarpMax(sharedMemT[threadIdx.x]);
			}
			__syncthreads();

			//First thread in block will now atomic add the result
			if (threadIdx.x == 0)
			{
				if constexpr (std::is_integral<T>::value && std::is_signed<T>::value)
				{
					using unsigned_T = typename std::make_unsigned<T>::type;
					atomicMax(reinterpret_cast<unsigned_T*>(&output[blockIdx.y]), (unsigned_T)blockSum);
				}
				else
				{
					atomicMax(&output[blockIdx.y], blockSum);
				}
			}
		}
	}

	template<typename T>
	void tensorMax(const tensor<T>& input, tensor<T>& output, const uint32_t maxDimIdx)
	{
		if constexpr (sizeof(T) < 4 || std::is_integral<T>::value && std::is_signed<T>::value)
		{
			throw std::runtime_error("Sum is currently not supported for that tensor type.");
		}
		else
		{
			uint32_t maxElementStride = 1;
			for (size_t i = maxDimIdx + 1; i < input.getDimensions().size(); i++)
			{
				maxElementStride *= input.getDimensions()[i].dim;
			}

			const uint32_t maxDim = input.getDimensions()[maxDimIdx].dim;
			const uint32_t dimsToMax = output.elementCount();
			const uint32_t blocksMade = std::min(dimsToMax, 40u);

			const dim3 blockDim(THREADS_PER_BLOCK);
			const dim3 gridDim(integerCeilDivision(maxDim, blockDim.x), blocksMade);
			if (autoGraph::isRecordingGraph())
			{
				{
					const void* outputPtr = reinterpret_cast<void*>(output.getGPUArray().begin());
					autoGraph::addMemsetNode(outputPtr, outputPtr, output.getGPUArray(), 0);
				}
				output = output + std::numeric_limits<T>::lowest();

				const std::vector<void*> inputPtrs = { reinterpret_cast<void*>(input.getGPUArray().begin()), reinterpret_cast<void*>(output.getGPUArray().begin()) };
				const void* outputPtr = reinterpret_cast<void*>(output.getGPUArray().begin());
				autoGraph::addKernelNode(inputPtrs, outputPtr, maxKernel<T>, blockDim, gridDim, (uint32_t)sizeof(T) * WARPS_PER_BLOCK, input.getGPUArrayConst(), output.getGPUArray(), maxElementStride, maxDim, dimsToMax, blocksMade);
			}
			else
			{
				cudaMemset(output.getGPUArray().begin(), 0, output.elementCount() * sizeof(T));
				output = output + std::numeric_limits<T>::lowest();
				cudabasic::executeKernel(maxKernel<T>, blockDim, gridDim, sizeof(T) * WARPS_PER_BLOCK, cuda::getDefaultStream(), input.getGPUArrayConst(), output.getGPUArray(), maxElementStride, maxDim, dimsToMax, blocksMade);
			}
		}
	}

	template void tensorMax(const tensor<bool>& input, tensor<bool>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<uint8_t>& input, tensor<uint8_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<uint16_t>& input, tensor<uint16_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<uint32_t>& input, tensor<uint32_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<uint64_t>& input, tensor<uint64_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<int8_t>& input, tensor<int8_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<int16_t>& input, tensor<int16_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<int32_t>& input, tensor<int32_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<int64_t>& input, tensor<int64_t>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<float>& input, tensor<float>& output, const uint32_t maxDimIdx);
	template void tensorMax(const tensor<double>& input, tensor<double>& output, const uint32_t maxDimIdx);
}