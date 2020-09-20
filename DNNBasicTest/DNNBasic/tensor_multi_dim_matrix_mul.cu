#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "tensor_multi_dim_matrix_mul.cuh"
#include "cudaBasics.h"
#include "matrix.h"
#include "tensor.h"
#include "kernel_tools.h"
#include "tensor_matrix_kernels.cu"

namespace dnnbasic
{
	using gpuArray = smallGPUArray<uint32_t, tensor<uint32_t>::MAX_DIMENSION_COUNT>;

	template <typename T>
	__global__ void multiDimMatrixMultiplication(
		const cudabasic::span<T> a, 
		const cudabasic::span<T> b, 
		cudabasic::span<T> c, 
		const uint32_t num_sub_blocks, 
		const uint32_t blockSize, 
		const gpuArray cSumDims,
		const gpuArray aDimStrides,
		const gpuArray bDimStrides,
		const gpuArray cDimStrides,
		const uint32_t aWidth,
		const uint32_t aHeight,
		const uint32_t bWidth,
		const uint32_t bHeight,
		const uint32_t aStride,
		const uint32_t bStride,
		const uint32_t cStride)
	{
		//fix this as we also have blockidx y
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx >= c.size())
		{
			return;
		}

		uint32_t index[gpuArray::MAX_LENGTH];
		uint32_t x = idx;
		// make x, y, z, .. indecies
		for (uint32_t i = 0; i < cSumDims.size(); i++)
		{
			index[i] = x / cSumDims[i];
			x = x % cSumDims[i];
		}

		// Convert to matricies
		uint32_t aMatrixIndex = 0;
		uint32_t bMatrixIndex = 0;
		uint32_t cMatrixIndex = 0;
		for (size_t i = 0; i < cSumDims.size(); i++)
		{
			aMatrixIndex += index[i] * aDimStrides[i];
			bMatrixIndex += index[i] * bDimStrides[i];
			cMatrixIndex += index[i] * cDimStrides[i];
		}

		const matrix<T> aMatrix(&a[aMatrixIndex], aWidth, aHeight, aStride);
		const matrix<T> bMatrix(&b[bMatrixIndex], bWidth, bHeight, aStride);
		matrix<T> cMatrix(&c[cMatrixIndex], aHeight, bWidth, cStride);

		dim3 blockOffset;
		dim3 threadOffset;

		const uint32_t widthIdx = cSumDims.size() - 1;
		const uint32_t heightIdx = cSumDims.size() - 2;
		blockOffset.x = index[widthIdx] / blockSize;
		blockOffset.y = index[heightIdx] / blockSize;
		threadOffset.x = index[widthIdx] % blockSize;
		threadOffset.y = index[heightIdx] % blockSize;

		matMul(aMatrix, bMatrix, cMatrix, num_sub_blocks, blockSize, blockOffset, threadOffset);
	}

	template <typename T>
	void tensorMultiDimMatMul(const tensor<T>& a, const tensor<T>& b, const tensor<T>& c)
	{

		std::vector<uint32_t> aDims(c.getDimensions.size());
		std::vector<uint32_t> bDims(c.getDimensions.size());
		std::vector<uint32_t> cDims(c.getDimensions.size());

		for (int i = c.getDimensions().size() - 1; i >= 0; i--)
		{
			if (i - a.getDimensions.size() < 0)
			{
				aDims[i] = 1;
			}
			else
			{
				aDims[i] = a.getDimensions()[i].dim;
			}
			if (i - b.getDimensions.size() < 0)
			{
				bDims[i] = 1;
			}
			else
			{
				bDims[i] = b.getDimensions()[i].dim;
			}
			cDims[i] = c.getDimensions()[i].dim;
		}

		gpuArray aStrides(aDims.size());
		gpuArray bStrides(bDims.size());
		gpuArray cStrides(cDims.size());

		for (uint32_t i = 0; i < cDims.size(); i++)
		{
			uint32_t aStride = 1;
			uint32_t bStride = 1;
			uint32_t cStride = 1;
			for (uint32_t g = i + 1; g < cDims.size(); g++)
			{
				aStride *= aDims[g];
				bStride *= bDims[g];
				cStride *= cDims[g];
			}
			aStrides[i] = aStride * (aDims[i] == 1 && bDims[i] != 1) ? 1 : 0;
			bStrides[i] = bStride * (bDims[i] == 1 && aDims[i] != 1) ? 1 : 0;
			cStrides[i] = cstride;
		}
		const uint32_t tensorWidth = 1;
		const uint32_t tensorHeight = 1;

		for (size_t i = 0; i < cDims.size(); i++)
		{
			if (i%2 == 0)
			{
				tensorHeight* = cDims[i];
			}
			else
			{
				tensorWidth* = cDims[i];
			}
		}

		const uint32_t blockSize = 32;
		const dim3 blockDim(blockSize, blockSize);
		const uint32_t sharedMemory = sizeof(T) * blockSize * blockSize * 2;
		const dim3 gridDim(integerCeilDivision(tensorWidth, blockDim.x), integerCeilDivision(tensorHeight, blockDim.y));
		const uint32_t num_sub_blocks = integerCeilDivision(left.getColumns(), blockSize);
		// kernel call?

	}
}