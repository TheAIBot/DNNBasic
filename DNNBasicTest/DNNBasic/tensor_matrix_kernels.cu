#include <cuda_runtime.h>
#include <array>
#include "tensor_matrix_kernels.cuh"
#include "cudaBasics.h"
#include "matrix.h"
#include "kernel_tools.h"
#include "cuda_settings.h"
#include "auto_graph.h"

namespace dnnbasic
{
	template <typename T>
	__global__ void matrixMultiplication(const matrix<T> a, const matrix<T> b, matrix<T> c, const uint32_t num_sub_blocks, const uint32_t blockSize)
	{
		// Block index
		const uint32_t bx = blockIdx.x;
		const uint32_t by = blockIdx.y;
		const uint32_t otx = threadIdx.x;
		const uint32_t oty = threadIdx.y;
		const uint32_t tx = threadIdx.x % blockSize;
		const uint32_t ty = threadIdx.y % blockSize;
		const uint32_t xSubBlockOffset = (threadIdx.x / blockSize) * blockSize;
		const uint32_t ySubBlockOffset = (threadIdx.y / blockSize) * blockSize;
		//Running sum of product of A and B matrices
		T Csub = 0;
		
		// need to fix shared memory offset for multidim matrix multiplication

		//Define some shared memory for a sub block of matrices A an B
		extern __shared__ __align__(sizeof(T)) int8_t sharedArray[];
		T* sharedMemT = reinterpret_cast<T*>(sharedArray);

		matrix<T> As(sharedMemT, blockDim.x, blockDim.y * 2);
		matrix<T> Bs(sharedMemT + As.size(), blockDim.x, blockDim.y * 2);

		{
			const uint32_t a_x = tx;
			const uint32_t a_y = ty + by * blockDim.y + ySubBlockOffset;
			const uint32_t b_x = tx + bx * blockDim.x + xSubBlockOffset;
			const uint32_t b_y = ty;

			//a submatrix can lie both inside and outside the bounds of the matrix.
			//We can't load any part that lies outside the bounds so instead 0 is
			//loaded into the submatrix because it doesn't change the result of
			//the sub matrix multiplication.
			As[oty][otx] = a.withinBounds(a_x, a_y) ? a[a_y][a_x] : (T)0;
			Bs[oty][otx] = b.withinBounds(b_x, b_y) ? b[b_y][b_x] : (T)0;
		}

		uint32_t shifter = 0;

		//iterate through the number of sub matrices of A and B
		for (uint32_t i = 0; i < num_sub_blocks; i++) {
			uint32_t oldShifter = shifter;

			if (i + 1 < num_sub_blocks)
			{
				shifter = ++shifter % 2;

				const uint32_t a_x = tx + (i + 1) * blockSize;
				const uint32_t a_y = ty + by * blockDim.y + ySubBlockOffset;
				const uint32_t b_x = tx + bx * blockDim.x + xSubBlockOffset;
				const uint32_t b_y = ty + (i + 1) * blockSize;

				//a submatrix can lie both inside and outside the bounds of the matrix.
				//We can't load any part that lies outside the bounds so instead 0 is
				//loaded into the submatrix because it doesn't change the result of
				//the sub matrix multiplication.
				As[oty + shifter * blockDim.y][otx] = a.withinBounds(a_x, a_y) ? a[a_y][a_x] : (T)0;
				Bs[oty + shifter * blockDim.y][otx] = b.withinBounds(b_x, b_y) ? b[b_y][b_x] : (T)0;
			}

			// change this so that we have min(a height, blocksize) <- is this valid?
			// Wait untill all threads have loaded their values into shared memory.
			__syncthreads();
			for (uint32_t k = 0; k < blockSize; ++k)
			{
				Csub += As[oty + oldShifter * blockDim.y][k + xSubBlockOffset] * Bs[k + ySubBlockOffset + oldShifter * blockDim.y][otx];
			}
			__syncthreads();

		}

		const uint32_t c_x = otx + bx * blockDim.x;
		const uint32_t c_y = oty + by * blockDim.y;

		// Write the resulting matrix multiplication into the result matrix if 
		// within bounds.
		if (!c.withinBounds(c_x, c_y))
		{
			return;
		}

		c[c_y][c_x] = Csub;
	}

	class blockConfig
	{
	private:
		uint32_t subBlockSize;
		uint32_t blocksWidth;
		uint32_t blocksHeight;

		template<typename T>
		uint32_t calcSuitability(const matrix<T>& mat, const uint32_t width, const uint32_t height) const
		{
			return std::min(mat.getColumns(), subBlockSize * width) * std::min(mat.getRows(), subBlockSize * height);
		}

	public:
		blockConfig(const uint32_t subBlockSize, const uint32_t width, const uint32_t height)
			: subBlockSize(subBlockSize), blocksWidth(width), blocksHeight(height)
		{ }

		template<typename T>
		uint32_t calcSuitability(const matrix<T>& left, const matrix<T>& right) const
		{
			return calcSuitability(left, blocksWidth, blocksHeight) + calcSuitability(right, blocksHeight, blocksWidth);
		}

		uint32_t getSubBlockSize() const
		{
			return subBlockSize;
		}

		dim3 getBlockDim() const
		{
			return dim3(subBlockSize * blocksWidth, subBlockSize * blocksHeight);
		}
	};

	const std::array<blockConfig, 4> blockConfigs =
	{
		blockConfig(32,  1,  1),
		blockConfig(16,  1,  4),
		blockConfig( 8,  1, 16),
		blockConfig( 4,  1, 64)
	};

	template <typename T>
	void tensorMatrixMulInternal(const matrix<T>& left, const matrix<T>& right, matrix<T>& result)
	{
		blockConfig bestConfig(0, 0, 0);
		uint32_t bestScore = 0;
		for (size_t i = 0; i < blockConfigs.size(); i++)
		{
			const uint32_t score = blockConfigs[i].calcSuitability(left, right);
			if (score > bestScore)
			{
				bestConfig = blockConfigs[i];
				bestScore = score;
			}
		}


		const uint32_t subBlockSize = bestConfig.getSubBlockSize();
		const dim3 blockDim = bestConfig.getBlockDim();
		const uint32_t sharedMemory = sizeof(T) * blockDim.x * blockDim.y * 2 * 2;
		const dim3 gridDim(integerCeilDivision(result.getColumns(), blockDim.x), integerCeilDivision(result.getRows(), blockDim.y));
		const uint32_t num_sub_blocks = integerCeilDivision(left.getColumns(), subBlockSize);
		
		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addKernelNode(matrixMultiplication<T>, blockDim, gridDim, sharedMemory, left, right, result, num_sub_blocks, subBlockSize);
		}
		else
		{
			cudabasic::executeKernel(matrixMultiplication<T>, blockDim, gridDim, sharedMemory, cuda::getDefaultStream(), left, right, result, num_sub_blocks, subBlockSize);
		}
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