#include <cuda_runtime.h>
#include "tensor_permute_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"

namespace dnnbasic
{
	template<typename T, uint32_t Max_Length>
	struct smallGPUArray
	{
		static constexpr uint32_t MAX_LENGTH = Max_Length;
		T arr[Max_Length];
		uint32_t length;

		smallGPUArray(uint32_t length) 
		{ 
			this->length = length; 
		}
		smallGPUArray(size_t length)
		{
			this->length = (uint32_t)length;
		}
		smallGPUArray(const std::vector<T>& copyFrom)
		{
			assert(copyFrom.size() < Max_Length);
			for (uint32_t i = 0; i < copyFrom.size(); i++)
			{
				arr[i] = copyFrom[i];
			}
			length = (uint32_t)copyFrom.size();
		}

		__device__ __host__ T& operator[](const uint32_t i)
		{
			assert(i < length);
			return arr[i];
		}

		__device__ __host__ T operator[](const uint32_t i) const
		{
			assert(i < length);
			return arr[i];
		}

		__device__ __host__ T& operator[](const std::size_t i)
		{
			assert(i < length);
			return arr[i];
		}

		__device__ __host__ T operator[](const std::size_t i) const
		{
			assert(i < length);
			return arr[i];
		}

		__device__ __host__ uint32_t size() const
		{
			return length;
		}
	};

	using dimsArray = smallGPUArray<uint32_t, tensor<uint32_t>::MAX_DIMENSION_COUNT>;

	template<typename T>
	__global__ void permute(const cudabasic::span<T> inData, cudabasic::span<T> outData, const dimsArray inSumDims, const dimsArray outSumDims, const dimsArray permuteIdxs)
	{
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx >= inData.size())
		{
			return;
		}

		uint32_t index[dimsArray::MAX_LENGTH];
		uint32_t x = idx;
		// make x, y, z, .. indecies
		for (uint32_t i = 0; i < inSumDims.size(); i++)
		{
			index[i] = x / inSumDims[i];
			x = x % inSumDims[i];
		}

		// make factors for indicies
		uint32_t outIndex = 0;
		for (uint32_t i = 0; i < inSumDims.size(); i++)
		{
			outIndex += index[permuteIdxs[i]] * outSumDims[i];
		}

		outData[outIndex] = inData[idx];
	}

	template <typename T>
	void tensorPermute(const tensor<T>& input, const tensor<T>& output, const std::vector<uint32_t>& dims)
	{
		const dim3 blockDim(256);
		const dim3 gridDim(integerCeilDivision(input.elementCount(), blockDim.x));

		dimsArray permutedIdx(dims);
		dimsArray inSumDims(dims.size());
		dimsArray outSumDims(dims.size());
		for (uint32_t i = 0; i < dims.size(); i++)
		{
			uint32_t inTotalDim = 1;
			uint32_t outTotalDim = 1;
			for (uint32_t g = i + 1; g < dims.size(); g++)
			{
				inTotalDim *= input.getDimensions()[g].dim;
				outTotalDim *= output.getDimensions()[g].dim;
			}
			inSumDims[i] = inTotalDim;
			outSumDims[i] = outTotalDim;
		}


		cudabasic::executeKernel(permute<T>, blockDim, gridDim, input.getGPUArrayConst(), output.getGPUArray(), inSumDims, outSumDims, permutedIdx);
	}

	template void tensorPermute(const tensor<bool>& input, const tensor<bool>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<uint8_t>& input, const tensor<uint8_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<uint16_t>& input, const tensor<uint16_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<uint32_t>& input, const tensor<uint32_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<uint64_t>& input, const tensor<uint64_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<int8_t>& input, const tensor<int8_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<int16_t>& input, const tensor<int16_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<int32_t>& input, const tensor<int32_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<int64_t>& input, const tensor<int64_t>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<float>& input, const tensor<float>& output, const std::vector<uint32_t>& dims);
	template void tensorPermute(const tensor<double>& input, const tensor<double>& output, const std::vector<uint32_t>& dims);
}