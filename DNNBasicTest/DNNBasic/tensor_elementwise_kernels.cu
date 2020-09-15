#include <cuda_runtime.h>
#include "tensor_elementwise_kernels.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"

namespace dnnbasic
{
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
	struct biArgElementWiseKernel
	{
		static void execute(const tensor<T>& left, const tensor<T>& right, const tensor<T>& result)
		{
			const dim3 blockDim(256);
			const dim3 gridDim(integerCeilDivision(result.elementCount(), blockDim.x));
			cudabasic::executeKernel(biArgElementWiseKernelSpanSpan<OP, T>, blockDim, gridDim, left.getGPUArray(), right.getGPUArray(), result.getGPUArray());
		}
		static void execute(const T left, const tensor<T>& right, const tensor<T>& result)
		{
			const dim3 blockDim(256);
			const dim3 gridDim(integerCeilDivision(result.elementCount(), blockDim.x));
			cudabasic::executeKernel(biArgElementWiseKernelScalarSpan<OP, T>, blockDim, gridDim, left, right.getGPUArray(), result.getGPUArray());
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

	void tensorMultiply(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result) { biArgElementWiseKernel<multiplyOp<bool>, bool>::execute(left, right, result); }
	void tensorMultiply(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result) { biArgElementWiseKernel<multiplyOp<uint8_t>, uint8_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result) { biArgElementWiseKernel<multiplyOp<uint16_t>, uint16_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result) { biArgElementWiseKernel<multiplyOp<uint32_t>, uint32_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result) { biArgElementWiseKernel<multiplyOp<uint64_t>, uint64_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result) { biArgElementWiseKernel<multiplyOp<int8_t>, int8_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result) { biArgElementWiseKernel<multiplyOp<int16_t>, int16_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result) { biArgElementWiseKernel<multiplyOp<int32_t>, int32_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result) { biArgElementWiseKernel<multiplyOp<int64_t>, int64_t>::execute(left, right, result); }
	void tensorMultiply(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result) { biArgElementWiseKernel<multiplyOp<float>, float>::execute(left, right, result); }
	void tensorMultiply(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result) { biArgElementWiseKernel<multiplyOp<double>, double>::execute(left, right, result); }

	void tensorMultiply(const bool left, const tensor<bool>& right, const tensor<bool>& result) { biArgElementWiseKernel<multiplyOp<bool>, bool>::execute(left, right, result); }
	void tensorMultiply(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result) { biArgElementWiseKernel<multiplyOp<uint8_t>, uint8_t>::execute(left, right, result); }
	void tensorMultiply(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result) { biArgElementWiseKernel<multiplyOp<uint16_t>, uint16_t>::execute(left, right, result); }
	void tensorMultiply(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result) { biArgElementWiseKernel<multiplyOp<uint32_t>, uint32_t>::execute(left, right, result); }
	void tensorMultiply(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result) { biArgElementWiseKernel<multiplyOp<uint64_t>, uint64_t>::execute(left, right, result); }
	void tensorMultiply(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result) { biArgElementWiseKernel<multiplyOp<int8_t>, int8_t>::execute(left, right, result); }
	void tensorMultiply(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result) { biArgElementWiseKernel<multiplyOp<int16_t>, int16_t>::execute(left, right, result); }
	void tensorMultiply(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result) { biArgElementWiseKernel<multiplyOp<int32_t>, int32_t>::execute(left, right, result); }
	void tensorMultiply(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result) { biArgElementWiseKernel<multiplyOp<int64_t>, int64_t>::execute(left, right, result); }
	void tensorMultiply(const float left, const tensor<float>& right, const tensor<float>& result) { biArgElementWiseKernel<multiplyOp<float>, float>::execute(left, right, result); }
	void tensorMultiply(const double left, const tensor<double>& right, const tensor<double>& result) { biArgElementWiseKernel<multiplyOp<double>, double>::execute(left, right, result); }

	void tensorAdd(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result) { biArgElementWiseKernel<addOp<bool>, bool>::execute(left, right, result); }
	void tensorAdd(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result) { biArgElementWiseKernel<addOp<uint8_t>, uint8_t>::execute(left, right, result); }
	void tensorAdd(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result) { biArgElementWiseKernel<addOp<uint16_t>, uint16_t>::execute(left, right, result); }
	void tensorAdd(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result) { biArgElementWiseKernel<addOp<uint32_t>, uint32_t>::execute(left, right, result); }
	void tensorAdd(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result) { biArgElementWiseKernel<addOp<uint64_t>, uint64_t>::execute(left, right, result); }
	void tensorAdd(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result) { biArgElementWiseKernel<addOp<int8_t>, int8_t>::execute(left, right, result); }
	void tensorAdd(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result) { biArgElementWiseKernel<addOp<int16_t>, int16_t>::execute(left, right, result); }
	void tensorAdd(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result) { biArgElementWiseKernel<addOp<int32_t>, int32_t>::execute(left, right, result); }
	void tensorAdd(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result) { biArgElementWiseKernel<addOp<int64_t>, int64_t>::execute(left, right, result); }
	void tensorAdd(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result) { biArgElementWiseKernel<addOp<float>, float>::execute(left, right, result); }
	void tensorAdd(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result) { biArgElementWiseKernel<addOp<double>, double>::execute(left, right, result); }

	void tensorAdd(const bool left, const tensor<bool>& right, const tensor<bool>& result) { biArgElementWiseKernel<addOp<bool>, bool>::execute(left, right, result); }
	void tensorAdd(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result) { biArgElementWiseKernel<addOp<uint8_t>, uint8_t>::execute(left, right, result); }
	void tensorAdd(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result) { biArgElementWiseKernel<addOp<uint16_t>, uint16_t>::execute(left, right, result); }
	void tensorAdd(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result) { biArgElementWiseKernel<addOp<uint32_t>, uint32_t>::execute(left, right, result); }
	void tensorAdd(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result) { biArgElementWiseKernel<addOp<uint64_t>, uint64_t>::execute(left, right, result); }
	void tensorAdd(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result) { biArgElementWiseKernel<addOp<int8_t>, int8_t>::execute(left, right, result); }
	void tensorAdd(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result) { biArgElementWiseKernel<addOp<int16_t>, int16_t>::execute(left, right, result); }
	void tensorAdd(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result) { biArgElementWiseKernel<addOp<int32_t>, int32_t>::execute(left, right, result); }
	void tensorAdd(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result) { biArgElementWiseKernel<addOp<int64_t>, int64_t>::execute(left, right, result); }
	void tensorAdd(const float left, const tensor<float>& right, const tensor<float>& result) { biArgElementWiseKernel<addOp<float>, float>::execute(left, right, result); }
	void tensorAdd(const double left, const tensor<double>& right, const tensor<double>& result) { biArgElementWiseKernel<addOp<double>, double>::execute(left, right, result); }

	void tensorSubtract(const tensor<bool>& left, const tensor<bool>& right, const tensor<bool>& result) { biArgElementWiseKernel<subtractOp<bool>, bool>::execute(left, right, result); }
	void tensorSubtract(const tensor<uint8_t>& left, const tensor<uint8_t>& right, const tensor<uint8_t>& result) { biArgElementWiseKernel<subtractOp<uint8_t>, uint8_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<uint16_t>& left, const tensor<uint16_t>& right, const tensor<uint16_t>& result) { biArgElementWiseKernel<subtractOp<uint16_t>, uint16_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<uint32_t>& left, const tensor<uint32_t>& right, const tensor<uint32_t>& result) { biArgElementWiseKernel<subtractOp<uint32_t>, uint32_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<uint64_t>& left, const tensor<uint64_t>& right, const tensor<uint64_t>& result) { biArgElementWiseKernel<subtractOp<uint64_t>, uint64_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<int8_t>& left, const tensor<int8_t>& right, const tensor<int8_t>& result) { biArgElementWiseKernel<subtractOp<int8_t>, int8_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<int16_t>& left, const tensor<int16_t>& right, const tensor<int16_t>& result) { biArgElementWiseKernel<subtractOp<int16_t>, int16_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<int32_t>& left, const tensor<int32_t>& right, const tensor<int32_t>& result) { biArgElementWiseKernel<subtractOp<int32_t>, int32_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<int64_t>& left, const tensor<int64_t>& right, const tensor<int64_t>& result) { biArgElementWiseKernel<subtractOp<int64_t>, int64_t>::execute(left, right, result); }
	void tensorSubtract(const tensor<float>& left, const tensor<float>& right, const tensor<float>& result) { biArgElementWiseKernel<subtractOp<float>, float>::execute(left, right, result); }
	void tensorSubtract(const tensor<double>& left, const tensor<double>& right, const tensor<double>& result) { biArgElementWiseKernel<subtractOp<double>, double>::execute(left, right, result); }

	void tensorSubtract(const bool left, const tensor<bool>& right, const tensor<bool>& result) { biArgElementWiseKernel<subtractOp<bool>, bool>::execute(left, right, result); }
	void tensorSubtract(const uint8_t left, const tensor<uint8_t>& right, const tensor<uint8_t>& result) { biArgElementWiseKernel<subtractOp<uint8_t>, uint8_t>::execute(left, right, result); }
	void tensorSubtract(const uint16_t left, const tensor<uint16_t>& right, const tensor<uint16_t>& result) { biArgElementWiseKernel<subtractOp<uint16_t>, uint16_t>::execute(left, right, result); }
	void tensorSubtract(const uint32_t left, const tensor<uint32_t>& right, const tensor<uint32_t>& result) { biArgElementWiseKernel<subtractOp<uint32_t>, uint32_t>::execute(left, right, result); }
	void tensorSubtract(const uint64_t left, const tensor<uint64_t>& right, const tensor<uint64_t>& result) { biArgElementWiseKernel<subtractOp<uint64_t>, uint64_t>::execute(left, right, result); }
	void tensorSubtract(const int8_t left, const tensor<int8_t>& right, const tensor<int8_t>& result) { biArgElementWiseKernel<subtractOp<int8_t>, int8_t>::execute(left, right, result); }
	void tensorSubtract(const int16_t left, const tensor<int16_t>& right, const tensor<int16_t>& result) { biArgElementWiseKernel<subtractOp<int16_t>, int16_t>::execute(left, right, result); }
	void tensorSubtract(const int32_t left, const tensor<int32_t>& right, const tensor<int32_t>& result) { biArgElementWiseKernel<subtractOp<int32_t>, int32_t>::execute(left, right, result); }
	void tensorSubtract(const int64_t left, const tensor<int64_t>& right, const tensor<int64_t>& result) { biArgElementWiseKernel<subtractOp<int64_t>, int64_t>::execute(left, right, result); }
	void tensorSubtract(const float left, const tensor<float>& right, const tensor<float>& result) { biArgElementWiseKernel<subtractOp<float>, float>::execute(left, right, result); }
	void tensorSubtract(const double left, const tensor<double>& right, const tensor<double>& result) { biArgElementWiseKernel<subtractOp<double>, double>::execute(left, right, result); }
}