#include <cuda_runtime.h>
#include "tensor_permute_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"

namespace dnnbasic
{
	template<typename T>
	struct tensorDims
	{
		uint32_t dims[10];
		uint32_t dimCount;

		tensorDims(const tensor<T>& inTensor)
		{
			auto copyDims = inTensor.getDimensions();
			for (uint32_t i = 0; i < copyDims.size(); i++)
			{
				dims[i] = copyDims[i].dim;
			}
			dimCount = (uint32_t)copyDims.size();
		}
	};

	struct permuteIndicies
	{
		uint32_t indicies[10];
		uint32_t indexCount;

		permuteIndicies(const std::vector<uint32_t>& inIndicies)
		{
			for (uint32_t i = 0; i < inIndicies.size(); i++)
			{
				indicies[i] = inIndicies[i];
			}
			indexCount = (uint32_t)inIndicies.size();
		}
	};

	template<typename T>
	__global__ void permute(const cudabasic::span<T> inData, cudabasic::span<T> outData, tensorDims<T> inDataDimension, tensorDims<T> outDataDimension, permuteIndicies permuteIdxs)
	{
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t index[10];

		uint32_t x = idx;
		// make x, y, z, .. indecies
		for (uint32_t i = 0; i < inDataDimension.dimCount; i++)
		{
			uint32_t totalDim = 1;
			for (uint32_t g = i + 1; g < inDataDimension.dimCount; g++)
			{
				totalDim *= inDataDimension.dims[g];
			}
			index[i] = x / totalDim;
			x = x % totalDim;
		}

		// make factors for indicies
		uint32_t inIndex = 0;
		uint32_t outIndex = 0;
		for (uint32_t i = 0; i < inDataDimension.dimCount; i++)
		{
			uint32_t totalDimIn = 1;
			uint32_t totalDimOut = 1;
			for (uint32_t g = i + 1; g < inDataDimension.dimCount; g++)
			{
				totalDimIn *= inDataDimension.dims[g];
				totalDimOut *= outDataDimension.dims[g];
			}
			inIndex += index[i] * totalDimIn;
			outIndex += index[permuteIdxs.indicies[i]] * totalDimOut;

		}

		if (inIndex >= inData.size() || outIndex >= outData.size())
		{
			return;
		}

		outData[outIndex] = inData[inIndex];
	}

	template <typename T>
	void tensorPermute(const tensor<T>& input, const tensor<T>& output, const std::vector<uint32_t>& dims)
	{
		const dim3 blockDim(256);
		const dim3 gridDim(integerCeilDivision(input.elementCount(), blockDim.x));

		cudabasic::executeKernel(permute<T>, blockDim, gridDim, input.getGPUArrayConst(), output.getGPUArray(), tensorDims<T>(input), tensorDims<T>(output), permuteIndicies(dims));
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