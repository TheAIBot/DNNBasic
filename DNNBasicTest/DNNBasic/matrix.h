#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include "span.h"

#include <stdio.h>

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

		__device__ __host__ cudabasic::span<T> operator[](const uint32_t rowIndex)
		{
			assert(rowIndex < rows);
			return cudabasic::span<T>(arr + rowIndex * columns, columns);
		}

		__device__ __host__ const cudabasic::span<T> operator[](const uint32_t rowIndex) const
		{
			assert(rowIndex < rows);
			return cudabasic::span<T>(arr + rowIndex * columns, columns);
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

		__device__ __host__ T* begin() const
		{
			return arr;
		}

		__device__ __host__ T* end() const
		{
			return arr + columns * rows;
		}
	};
}