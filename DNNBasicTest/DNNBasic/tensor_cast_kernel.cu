#include <cuda_runtime.h>
#include <cstdint>
#include "tensor_cast_kernel.cuh"
#include "kernel_tools.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "auto_graph.h"

namespace dnnbasic
{
	template<typename From, typename To>
	__global__ void cast(const cudabasic::span<From> from, cudabasic::span<To> to)
	{
		const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= from.size())
		{
			return;
		}

		if (std::is_floating_point<From>::value && std::is_unsigned<To>::value)
		{
			to[index] = (To)(int64_t)from[index];
		}
		else
		{
			to[index] = (To)from[index];
		}
	}

	template<typename From, typename To>
	void tensorCast(const tensor<From>& from, tensor<To>& to)
	{
		const dim3 blockDim(256);
		const dim3 gridDim(integerCeilDivision(from.elementCount(), blockDim.x));
		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addKernelNode(cast<From, To>, blockDim, gridDim, 0, from.getGPUArrayConst(), to.getGPUArray());
		}
		else
		{
			cudabasic::executeKernel(cast<From, To>, blockDim, gridDim, 0, cuda::getDefaultStream(), from.getGPUArrayConst(), to.getGPUArray());
		}
	}

#define CAST_FROM_TO(fromTyp, toTyp) \
	template void tensorCast(const tensor<fromTyp>& from, tensor<toTyp>& to);

#define CAST_FROM(fromTyp) \
	CAST_FROM_TO(fromTyp, bool) \
	CAST_FROM_TO(fromTyp, uint8_t) \
	CAST_FROM_TO(fromTyp, uint16_t) \
	CAST_FROM_TO(fromTyp, uint32_t) \
	CAST_FROM_TO(fromTyp, uint64_t) \
	CAST_FROM_TO(fromTyp, int8_t) \
	CAST_FROM_TO(fromTyp, int16_t) \
	CAST_FROM_TO(fromTyp, int32_t) \
	CAST_FROM_TO(fromTyp, int64_t) \
	CAST_FROM_TO(fromTyp, float) \
	CAST_FROM_TO(fromTyp, double)

	CAST_FROM(bool)
	CAST_FROM(uint8_t)
	CAST_FROM(uint16_t)
	CAST_FROM(uint32_t)
	CAST_FROM(uint64_t)
	CAST_FROM(int8_t)
	CAST_FROM(int16_t)
	CAST_FROM(int32_t)
	CAST_FROM(int64_t)
	CAST_FROM(float)
	CAST_FROM(double)
}