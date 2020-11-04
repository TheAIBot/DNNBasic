#include <cuda_runtime.h>
#include "tensor_elementwise_kernels.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

namespace dnnbasic
{
	using gpuArray = smallGPUArray<uint32_t, tensor<uint32_t>::MAX_DIMENSION_COUNT>;

	template<typename OP, typename T>
	__global__ void biArgElementWiseKernelSpanSpanBroadcast(
		const cudabasic::span<T> left,
		const cudabasic::span<T> right,
		cudabasic::span<T> output,
		const gpuArray leftStrides,
		const gpuArray rightStrides,
		const gpuArray outputStrides)
	{
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx >= output.size())
		{
			return;
		}

		uint32_t leftIndex = 0;
		uint32_t rightIndex = 0;
		uint32_t x = idx;
		for (uint32_t i = 0; i < outputStrides.size(); i++)
		{
			const uint32_t index = x / outputStrides[i];
			leftIndex += index * leftStrides[i];
			rightIndex += index * rightStrides[i];

			x = x % outputStrides[i];
		}

		output[idx] = OP::operation(left[leftIndex], right[rightIndex]);
	}
	template<typename OP, typename T>
	__global__ void biArgElementWiseKernelSpanSpan(const cudabasic::span<T> left, const cudabasic::span<T> right, cudabasic::span<T> output)
	{
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= output.size())
		{
			return;
		}
		output[index] = OP::operation(left[index], right[index]);
	}
	template<typename OP, typename T>
	__global__ void biArgElementWiseKernelScalarSpan(const T left, const cudabasic::span<T> right, cudabasic::span<T> output)
	{
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= output.size())
		{
			return;
		}
		output[index] = OP::operation(left, right[index]);
	}
	template<typename OP, typename T>
	__global__ void biArgElementWiseKernelScalarSpan(const cudabasic::span<T> left, const T right, cudabasic::span<T> output)
	{
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= output.size())
		{
			return;
		}
		output[index] = OP::operation(left[index], right);
	}

	template<typename OP, typename T>
	struct biArgElementWiseKernel
	{
		static void execute(const tensor<T>& left, const tensor<T>& right, const tensor<T>& result, const bool isBroadcasted)
		{
			if (isBroadcasted)
			{
				std::vector<uint32_t> aDims(result.getDimensions().size());
				std::vector<uint32_t> bDims(result.getDimensions().size());
				std::vector<uint32_t> cDims(result.getDimensions().size());

				int32_t aDimsIdx = (int32_t)left.getDimensions().size() - 1;
				int32_t bDimsIdx = (int32_t)right.getDimensions().size() - 1;
				int32_t cDimsIdx = (int32_t)result.getDimensions().size() - 1;

				// Convert aDims and bDims into a shape tensor in which length of the tensor is
				// the same size as the output c. The ideas is to perform an internal broadcasting of a and b
				// such that these can be multiplied.
				for (int32_t i = (int32_t)result.getDimensions().size() - 1; i >= 0; i--)
				{
					if (aDimsIdx < 0)
					{
						aDims[i] = 1;
					}
					else
					{
						aDims[i] = left.getDimensions()[aDimsIdx].dim;
						aDimsIdx--;
					}
					if (bDimsIdx < 0)
					{
						bDims[i] = 1;
					}
					else
					{
						bDims[i] = right.getDimensions()[bDimsIdx].dim;
						bDimsIdx--;
					}
					cDims[i] = result.getDimensions()[i].dim;
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

				const dim3 blockDim(256);
				const dim3 gridDim(integerCeilDivision(result.elementCount(), blockDim.x));
				if (autoGraph::isRecordingGraph())
				{
					const std::vector<void*> inputs = { reinterpret_cast<void*>(left.getGPUArray().begin()), reinterpret_cast<void*>(right.getGPUArray().begin()) };
					const void* output = reinterpret_cast<void*>(result.getGPUArray().begin());
					autoGraph::addKernelNode(inputs, output, biArgElementWiseKernelSpanSpanBroadcast<OP, T>, blockDim, gridDim, 0u, left.getGPUArray(), right.getGPUArray(), result.getGPUArray(), aStrides, bStrides, cStrides);
				}
				else
				{
					cudabasic::executeKernel(biArgElementWiseKernelSpanSpanBroadcast<OP, T>, blockDim, gridDim, 0, cuda::getDefaultStream(), left.getGPUArray(), right.getGPUArray(), result.getGPUArray(), aStrides, bStrides, cStrides);
				}
			}
			else
			{
				const dim3 blockDim(256);
				const dim3 gridDim(integerCeilDivision(result.elementCount(), blockDim.x));
				if (autoGraph::isRecordingGraph())
				{
					const std::vector<void*> inputs = { reinterpret_cast<void*>(left.getGPUArray().begin()), reinterpret_cast<void*>(right.getGPUArray().begin()) };
					const void* output = reinterpret_cast<void*>(result.getGPUArray().begin());
					autoGraph::addKernelNode(inputs, output, biArgElementWiseKernelSpanSpan<OP, T>, blockDim, gridDim, 0u, left.getGPUArray(), right.getGPUArray(), result.getGPUArray());
				}
				else
				{
					cudabasic::executeKernel(biArgElementWiseKernelSpanSpan<OP, T>, blockDim, gridDim, 0, cuda::getDefaultStream(), left.getGPUArray(), right.getGPUArray(), result.getGPUArray());
				}
			}
		}
		static void execute(const T left, const tensor<T>& right, const tensor<T>& result, const bool isBroadcasted)
		{
			const dim3 blockDim(256);
			const dim3 gridDim(integerCeilDivision(result.elementCount(), blockDim.x));
			if (autoGraph::isRecordingGraph())
			{
				const std::vector<void*> inputs = { reinterpret_cast<void*>(right.getGPUArray().begin()) };
				const void* output = reinterpret_cast<void*>(result.getGPUArray().begin());
				autoGraph::addKernelNode(inputs, output, biArgElementWiseKernelScalarSpan<OP, T>, blockDim, gridDim, 0u, left, right.getGPUArray(), result.getGPUArray());
			}
			else
			{
				cudabasic::executeKernel(biArgElementWiseKernelScalarSpan<OP, T>, blockDim, gridDim, 0, cuda::getDefaultStream(), left, right.getGPUArray(), result.getGPUArray());
			}
		}
		static void execute(const tensor<T>& left, const T right, const tensor<T>& result, const bool isBroadcasted)
		{
			const dim3 blockDim(256);
			const dim3 gridDim(integerCeilDivision(result.elementCount(), blockDim.x));
			if (autoGraph::isRecordingGraph())
			{
				const std::vector<void*> inputs = { reinterpret_cast<void*>(left.getGPUArray().begin()) };
				const void* output = reinterpret_cast<void*>(result.getGPUArray().begin());
				autoGraph::addKernelNode(inputs, output, biArgElementWiseKernelScalarSpan<OP, T>, blockDim, gridDim, 0u, left.getGPUArray(), right, result.getGPUArray());
			}
			else
			{
				cudabasic::executeKernel(biArgElementWiseKernelScalarSpan<OP, T>, blockDim, gridDim, 0, cuda::getDefaultStream(), left.getGPUArray(), right, result.getGPUArray());
			}
		}
	};

	template<typename T>
	struct multiplyOp
	{
		static __device__ T operation(const T left, const T right)
		{
			return left * right;
		}
	};
	template<typename T>
	struct divideOp
	{
		static __device__ T operation(const T left, const T right)
		{
			return left / right;
		}
	};
	template<typename T>
	struct addOp
	{
		static __device__ T operation(const T left, const T right)
		{
			return left + right;
		}
	};
	template<typename T>
	struct subtractOp
	{
		static __device__ T operation(const T left, const T right)
		{
			return left - right;
		}
	};

	void tensorMultiply(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<bool>, bool>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint8_t>, uint8_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint16_t>, uint16_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint32_t>, uint32_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint64_t>, uint64_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int8_t>, int8_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int16_t>, int16_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int32_t>, int32_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int64_t>, int64_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<float>, float>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<double>, double>::execute(left, right, result, isBroadcasted); }

	void tensorMultiply(const bool left, const tensor<bool>& right, const tensor<bool>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<bool>, bool>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint8_t>, uint8_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint16_t>, uint16_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint32_t>, uint32_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<uint64_t>, uint64_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int8_t>, int8_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int16_t>, int16_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int32_t>, int32_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<int64_t>, int64_t>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const float left, const tensor<float>& right, const tensor<float>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<float>, float>::execute(left, right, result, isBroadcasted); }
	void tensorMultiply(const double left, const tensor<double>& right, const tensor<double>& result, const bool isBroadcasted) { biArgElementWiseKernel<multiplyOp<double>, double>::execute(left, right, result, isBroadcasted); }

	void tensorDiv(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result) { biArgElementWiseKernel<divideOp<bool>, bool>::execute(left, right, result, false); }
	void tensorDiv(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result) { biArgElementWiseKernel<divideOp<uint8_t>, uint8_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result) { biArgElementWiseKernel<divideOp<uint16_t>, uint16_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result) { biArgElementWiseKernel<divideOp<uint32_t>, uint32_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result) { biArgElementWiseKernel<divideOp<uint64_t>, uint64_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result) { biArgElementWiseKernel<divideOp<int8_t>, int8_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result) { biArgElementWiseKernel<divideOp<int16_t>, int16_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result) { biArgElementWiseKernel<divideOp<int32_t>, int32_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result) { biArgElementWiseKernel<divideOp<int64_t>, int64_t>::execute(left, right, result, false); }
	void tensorDiv(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result) { biArgElementWiseKernel<divideOp<float>, float>::execute(left, right, result, false); }
	void tensorDiv(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result) { biArgElementWiseKernel<divideOp<double>, double>::execute(left, right, result, false); }

	void tensorDiv(const tensor<bool>& left, const bool right, const tensor<bool>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<bool>, bool>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<uint8_t>& left, const uint8_t right, const tensor<uint8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<uint8_t>, uint8_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<uint16_t>& left, const uint16_t right, const tensor<uint16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<uint16_t>, uint16_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<uint32_t>& left, const uint32_t right, const tensor<uint32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<uint32_t>, uint32_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<uint64_t>& left, const uint64_t right, const tensor<uint64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<uint64_t>, uint64_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<int8_t>& left, const int8_t right, const tensor<int8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<int8_t>, int8_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<int16_t>& left, const int16_t right, const tensor<int16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<int16_t>, int16_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<int32_t>& left, const int32_t right, const tensor<int32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<int32_t>, int32_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<int64_t>& left, const int64_t right, const tensor<int64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<int64_t>, int64_t>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<float>& left, const float right, const tensor<float>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<float>, float>::execute(left, right, result, isBroadcasted); }
	void tensorDiv(const tensor<double>& left, const double right, const tensor<double>& result, const bool isBroadcasted) { biArgElementWiseKernel<divideOp<double>, double>::execute(left, right, result, isBroadcasted); }

	void tensorAdd(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<bool>, bool>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint8_t>, uint8_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint16_t>, uint16_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint32_t>, uint32_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint64_t>, uint64_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int8_t>, int8_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int16_t>, int16_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int32_t>, int32_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int64_t>, int64_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<float>, float>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<double>, double>::execute(left, right, result, isBroadcasted); }

	void tensorAdd(const bool left, const tensor<bool>& right, const tensor<bool>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<bool>, bool>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint8_t>, uint8_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint16_t>, uint16_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint32_t>, uint32_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<uint64_t>, uint64_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int8_t>, int8_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int16_t>, int16_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int32_t>, int32_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<int64_t>, int64_t>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const float left, const tensor<float>& right, const tensor<float>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<float>, float>::execute(left, right, result, isBroadcasted); }
	void tensorAdd(const double left, const tensor<double>& right, const tensor<double>& result, const bool isBroadcasted) { biArgElementWiseKernel<addOp<double>, double>::execute(left, right, result, isBroadcasted); }

	void tensorSubtract(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<bool>, bool>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint8_t>, uint8_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint16_t>, uint16_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint32_t>, uint32_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint64_t>, uint64_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int8_t>, int8_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int16_t>, int16_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int32_t>, int32_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int64_t>, int64_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<float>, float>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<double>, double>::execute(left, right, result, isBroadcasted); }

	void tensorSubtract(const bool left, const tensor<bool>& right, const tensor<bool>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<bool>, bool>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint8_t>, uint8_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint16_t>, uint16_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint32_t>, uint32_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<uint64_t>, uint64_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int8_t>, int8_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int16_t>, int16_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int32_t>, int32_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<int64_t>, int64_t>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const float left, const tensor<float>& right, const tensor<float>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<float>, float>::execute(left, right, result, isBroadcasted); }
	void tensorSubtract(const double left, const tensor<double>& right, const tensor<double>& result, const bool isBroadcasted) { biArgElementWiseKernel<subtractOp<double>, double>::execute(left, right, result, isBroadcasted); }
}