#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

namespace dnnbasic
{
	template<typename T>
	class matrix
	{
	private:
		T* arr;
		uint32_t columns;
		uint32_t rows;

	public:
		__device__ __host__ matrix(T* arrPtr, uint32_t columns, uint32_t rows)
		{
			assert(arrPtr != nullptr);
			this->arr = arrPtr;
			this->columns = columns;
			this->rows = rows;
		}

		__device__ __host__ T* operator[](const uint32_t rowIndex)
		{
			assert(rowIndex < rows);
			return &arr[rowIndex * columns];
		}

		__device__ __host__ const T* operator[](const uint32_t rowIndex) const
		{
			assert(rowIndex < rows);
			return &arr[rowIndex * columns];
		}

		__device__ __host__ uint32_t getColumns() const
		{
			return columns;
		}

		__device__ __host__ uint32_t getRows() const
		{
			return rows;
		}

		__device__ __host__ bool withinBounds(const uint32_t column, const uint32_t row) const
		{
			return column < this->columns&& row < this->rows;
		}
	};
}