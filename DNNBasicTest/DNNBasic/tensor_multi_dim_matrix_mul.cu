#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "tensor_multi_dim_matrix_mul.cuh"
#include "cudaBasics.h"
#include "matrix.h"
#include "tensor.h"
#include "kernel_tools.h"
#include "tensor_matrix_kernels.cuh"

namespace dnnbasic
{
	using gpuArray = smallGPUArray<uint32_t, tensor<uint32_t>::MAX_DIMENSION_COUNT>;

	template <typename T>
	__device__ T max(const T a, const T b) 
	{
		return a > b ? a : b;
	}

	template <typename T>
	__device__ T min(const T a, const T b)
	{
		return a < b ? a : b;
	}

	//template <typename T>
	//__device__ void matMulInternal(const matrix<T> a, const matrix<T> b, matrix<T> c, const uint32_t num_sub_blocks, const uint32_t blockSizeX, const uint32_t blockSizeY, const dim3 blockOffset, const dim3 threadOffset, const uint32_t sharedOffset)
	//{
	//	// Block index
	//	const uint32_t bx = blockOffset.x;
	//	const uint32_t by = blockOffset.y;
	//	const uint32_t tx = threadOffset.x;
	//	const uint32_t ty = threadOffset.y;
	//	//Running sum of product of A and B matrices
	//	T Csub = 0;

	//	// need to fix shared memory offset for multidim matrix multiplication

	//	//Define some shared memory for a sub block of matrices A an B
	//	extern __shared__ __align__(sizeof(T)) int8_t sharedArray[];
	//	T* sharedMemT = reinterpret_cast<T*>(sharedArray);

	//	matrix<T> As(sharedMemT + sharedOffset, blockSizeX, blockSizeY);
	//	matrix<T> Bs(sharedMemT + sharedOffset + blockSizeX * blockSizeY, blockSizeX, blockSizeY);

	//	//iterate through the number of sub matrices of A and B
	//	for (uint32_t i = 0; i < num_sub_blocks; i++) {
	//		const uint32_t a_x = tx + i * blockSizeX;
	//		const uint32_t a_y = ty + by * blockSizeY;
	//		const uint32_t b_x = tx + bx * blockSizeX;
	//		const uint32_t b_y = ty + i * blockSizeY;

	//		//a submatrix can lie both inside and outside the bounds of the matrix.
	//		//We can't load any part that lies outside the bounds so instead 0 is
	//		//loaded into the submatrix because it doesn't change the result of
	//		//the sub matrix multiplication.
	//		As[ty][tx] = a.withinBounds(a_x, a_y) ? a[a_y][a_x] : (T)0;
	//		Bs[ty][tx] = b.withinBounds(b_x, b_y) ? b[b_y][b_x] : (T)0;

	//		// change this so that we have min(a height, blocksize) <- is this valid?
	//		// Wait untill all threads have loaded their values into shared memory.
	//		__syncthreads();
	//		for (uint32_t k = 0; k < blockSizeX; ++k)
	//		{
	//			Csub += As[ty][k] * Bs[k][tx];
	//		}
	//		__syncthreads();

	//	}

	//	const uint32_t c_x = tx + bx * blockSizeX;
	//	const uint32_t c_y = ty + by * blockSizeY;

	//	// Write the resulting matrix multiplication into the result matrix if 
	//	// within bounds.
	//	if (!c.withinBounds(c_x, c_y))
	//	{
	//		return;
	//	}

	//	c[c_y][c_x] = Csub;
	//}

	//template <typename T>
	//__global__ void matrixMultiplication(const matrix<T> a, const matrix<T> b, matrix<T> c, const uint32_t num_sub_blocks, const uint32_t blockSize)
	//{
	//	matMulInternal(a, b, c, num_sub_blocks, blockSize, blockSize, blockIdx, threadIdx, 0);
	//}

	__device__ int icd(int a, int b)
	{
		//return (int) math.ceil((float)a / b);
		return (a + (b - 1)) / b;
	}

	template <typename T>
	__global__ void multiDimMatrixMultiplication(
		const cudabasic::span<T> a, 
		const cudabasic::span<T> b, 
		cudabasic::span<T> c,
		const gpuArray aDimStrides,
		const gpuArray bDimStrides,
		const gpuArray cDimStrides,
		const uint32_t aWidth,
		const uint32_t aHeight,
		const uint32_t bWidth,
		const uint32_t bHeight,
		const uint32_t num_sub_blocks)
	{
		const uint32_t cMatrixWidth = bWidth;
		const uint32_t cMatrixHeight = aHeight;

		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

		const uint32_t blockSize = min(32u, max(cMatrixWidth, cMatrixHeight));
		const dim3 blockDimq(blockSize, blockSize);
		const dim3 gridDimq(icd(cMatrixWidth, blockDimq.x), icd(cMatrixHeight, blockDimq.y));
		const uint32_t sharedMemory = sizeof(T) * blockDimq.x * blockDimq.y * 2;

		for (size_t qweqwe = 0; ; qweqwe++)
		{
			const uint32_t idx = (threadIdx.x + qweqwe * blockDim.x) * (cMatrixWidth * cMatrixHeight);

			if (idx >= c.size())
			{
				return;
			}


			const uint32_t matrixDimsCount = 2;
			uint32_t index[gpuArray::MAX_LENGTH];
			uint32_t x = idx;
			// make x, y, z, .. indecies
			for (uint32_t i = 0; i < cDimStrides.size(); i++)
			{
				index[i] = x / cDimStrides[i];
				x = x % cDimStrides[i];
			}

			// Convert to matricies
			uint32_t aMatrixIndex = 0;
			uint32_t bMatrixIndex = 0;
			const uint32_t cMatrixIndex = idx;
			for (size_t i = 0; i < cDimStrides.size() - matrixDimsCount; i++)
			{
				aMatrixIndex += index[i] * aDimStrides[i];
				bMatrixIndex += index[i] * bDimStrides[i];
			}

			const matrix<T> aMatrix(&a[aMatrixIndex], aWidth, aHeight);
			const matrix<T> bMatrix(&b[bMatrixIndex], bWidth, bHeight);
			matrix<T> cMatrix(&c[cMatrixIndex], cMatrixWidth, cMatrixHeight);

			matrixMultiplication << <gridDimq, blockDimq, sharedMemory, stream >> > (aMatrix, bMatrix, cMatrix, num_sub_blocks, blockDimq.x);
		}

		cudaStreamDestroy(stream);
	}

	template <typename T>
	void tensorMultiDimMatMul(const tensor<T>& a, const tensor<T>& b, const tensor<T>& c)
	{

		std::vector<uint32_t> aDims(c.getDimensions().size());
		std::vector<uint32_t> bDims(c.getDimensions().size());
		std::vector<uint32_t> cDims(c.getDimensions().size());

		int32_t aDimsIdx = (int32_t)a.getDimensions().size() - 1;
		int32_t bDimsIdx = (int32_t)b.getDimensions().size() - 1;
		int32_t cDimsIdx = (int32_t)c.getDimensions().size() - 1;

		// Convert aDims and bDims into a shape tensor in which length of the tensor is
		// the same size as the output c. The ideas is to perform an internal broadcasting of a and b
		// such that these can be multiplied.
		for (int32_t i = (int32_t)c.getDimensions().size() - 1; i >= 0; i--)
		{
			if (aDimsIdx < 0)
			{
				aDims[i] = 1;
			}
			else
			{
				aDims[i] = a.getDimensions()[aDimsIdx].dim;
				aDimsIdx--;
			}
			if (bDimsIdx < 0)
			{
				bDims[i] = 1;
			}
			else
			{
				bDims[i] = b.getDimensions()[bDimsIdx].dim;
				bDimsIdx--;
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

			// To get the correct stride when using an array we multiply the following dimensions
			// together such that they correspond to accessing index i of the corresponding matrix 
			// with similar dimensions
			for (uint32_t g = i + 1; g < cDims.size(); g++)
			{
				aStride *= aDims[g];
				bStride *= bDims[g];
				cStride *= cDims[g];
			}
			// if dimension is broadcasted then the stride should be 0 to reuse the same matrix again
			aStrides[i] = aStride * ((aDims[i] == 1 && bDims[i] != 1) ? 0 : 1);
			bStrides[i] = bStride * ((bDims[i] == 1 && aDims[i] != 1) ? 0 : 1);
			cStrides[i] = cStride;
		}

		// height and width of the matrix
		const uint32_t aWidth = aDims[aDims.size() - 1];
		const uint32_t bWidth = bDims[bDims.size() - 1];
		const uint32_t aHeight = aDims[aDims.size() - 2];
		const uint32_t bHeight = bDims[bDims.size() - 2];

		const uint32_t blockSize = 784;
		const dim3 blockDim(blockSize);
		const dim3 gridDim(1);
		const uint32_t num_sub_blocks = integerCeilDivision(aWidth, blockSize);

		cudabasic::executeKernel(multiDimMatrixMultiplication<T>, blockDim, gridDim, a.getGPUArrayConst(), b.getGPUArrayConst(), c.getGPUArray(),
			aStrides, bStrides, cStrides, aWidth, aHeight, bWidth, bHeight, num_sub_blocks);
	}
	void tensorMultiDimMatrixMul(const tensor<bool>& a, const tensor<bool>& b, const tensor<bool>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<uint8_t>& a, const tensor<uint8_t>& b, tensor<uint8_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<uint16_t>& a, const tensor<uint16_t>& b, tensor<uint16_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<uint32_t>& a, const tensor<uint32_t>& b, tensor<uint32_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<uint64_t>& a, const tensor<uint64_t>& b, tensor<uint64_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<int8_t>& a, const tensor<int8_t>& b, tensor<int8_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<int16_t>& a, const tensor<int16_t>& b, tensor<int16_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<int32_t>& a, const tensor<int32_t>& b, tensor<int32_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<int64_t>& a, const tensor<int64_t>& b, tensor<int64_t>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<float>& a, const tensor<float>& b, tensor<float>& c) { tensorMultiDimMatMul(a, b, c); }
	void tensorMultiDimMatrixMul(const tensor<double>& a, const tensor<double>& b, tensor<double>& c) { tensorMultiDimMatMul(a, b, c); }
}