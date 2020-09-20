#include <cuda_runtime.h>
#include "tensor_matrix_kernels.cuh"
#include "cudaBasics.h"
#include "matrix.h"
#include "kernel_tools.h"

namespace dnnbasic
{
	template <typename T>
	__device__ void matMul(const matrix<T> a, const matrix<T> b, matrix<T> c, const uint32_t num_sub_blocks, const uint32_t blockSize, dim3 blockOffset, dim3 threadOffset)
	{
		// Block index
		const uint32_t bx = blockOffset.x;
		const uint32_t by = blockOffset.y;
		const uint32_t tx = threadOffset.x;
		const uint32_t ty = threadOffset.y;
		//Running sum of product of A and B matrices
		T Csub = 0;
		
		// need to fix shared memory offset for multidim matrix multiplication

		//Define some shared memory for a sub block of matrices A an B
		extern __shared__ __align__(sizeof(T)) int8_t sharedArray[];
		T* sharedMemT = reinterpret_cast<T*>(sharedArray);

		matrix<T> As(sharedMemT, blockSize, blockSize);
		matrix<T> Bs(sharedMemT + blockSize * blockSize, blockSize, blockSize);

		//iterate through the number of sub matrices of A and B
		for (uint32_t i = 0; i < num_sub_blocks; i++) {
			const uint32_t a_x = tx + i * blockSize;
			const uint32_t a_y = ty + by * blockSize;
			const uint32_t b_x = tx + bx * blockSize;
			const uint32_t b_y = ty + i * blockSize;

			//a submatrix can lie both inside and outside the bounds of the matrix.
			//We can't load any part that lies outside the bounds so instead 0 is
			//loaded into the submatrix because it doesn't change the result of
			//the sub matrix multiplication.
			As[ty][tx] = a.withinBounds(a_x, a_y) ? a[a_y][a_x] : (T)0;
			Bs[ty][tx] = b.withinBounds(b_x, b_y) ? b[b_y][b_x] : (T)0;

			// change this so that we have min(a height, blocksize) <- is this valid?
			// Wait untill all threads have loaded their values into shared memory.
			__syncthreads();
			for (uint32_t k = 0; k < blockSize; ++k)
			{
				Csub += As[ty][k] * Bs[k][tx];
			}
			__syncthreads();

		}

		const uint32_t c_x = tx + bx * blockSize;
		const uint32_t c_y = ty + by * blockSize;

		// Write the resulting matrix multiplication into the result matrix if 
		// within bounds.
		if (!c.withinBounds(c_x, c_y))
		{
			return;
		}

		c[c_y][c_x] = Csub;
	}

	template <typename T>
	__global__ void matrixMultiplication(const matrix<T> a, const matrix<T> b, matrix<T> c, const uint32_t num_sub_blocks, const uint32_t blockSize)
	{
		matMul(a, b, c, num_sub_blocks, blockSize, blockIdx, threadIdx);
	}


	template <typename T>
	void tensorMatrixMulInternal(const matrix<T>& left, const matrix<T>& right, matrix<T>& result)
	{
		const int matrixWidth = result.getColumns();
		const int matrixHeight = result.getRows();
		
		const uint32_t blockSize = 32; 
		const dim3 blockDim(blockSize, blockSize);
		const uint32_t sharedMemory = sizeof(T) * blockSize * blockSize * 2;
		const dim3 gridDim(integerCeilDivision(matrixWidth, blockDim.x), integerCeilDivision(matrixHeight, blockDim.y));
		const uint32_t num_sub_blocks = integerCeilDivision(left.getColumns(), blockSize);
		
		cudabasic::executeKernel(matrixMultiplication<T>, blockDim, gridDim, sharedMemory, left, right, result, num_sub_blocks, blockSize);
	}
	void tensorMatrixMul(const matrix<bool>& left, const matrix<bool>& right, matrix<bool>& result){tensorMatrixMulInternal(left, right, result);}
	void tensorMatrixMul(const matrix<uint8_t>& left, const matrix<uint8_t>& right, matrix<uint8_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<uint16_t>& left, const matrix<uint16_t>& right, matrix<uint16_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<uint32_t>& left, const matrix<uint32_t>& right, matrix<uint32_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<uint64_t>& left, const matrix<uint64_t>& right, matrix<uint64_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<int8_t>& left, const matrix<int8_t>& right, matrix<int8_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<int16_t>& left, const matrix<int16_t>& right, matrix<int16_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<int32_t>& left, const matrix<int32_t>& right, matrix<int32_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<int64_t>& left, const matrix<int64_t>& right, matrix<int64_t>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<float>& left, const matrix<float>& right, matrix<float>& result) { tensorMatrixMulInternal(left, right, result); }
	void tensorMatrixMul(const matrix<double>& left, const matrix<double>& right, matrix<double>& result) { tensorMatrixMulInternal(left, right, result); }
}