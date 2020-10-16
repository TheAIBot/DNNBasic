#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>

namespace dnnbasic
{
	/// <summary>
	/// If there is a remainder to the division then it adds 1 to the division result
	/// </summary>
	/// <param name="a">numerator</param>
	/// <param name="b">denominator</param>
	/// <returns></returns>
	int integerCeilDivision(int a, int b);

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
}