#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <numeric>
#include "span.h"
#include "gpuArray.h"
#include "tensor_matrix_kernels.cuh"

namespace dnnbasic
{
	struct namedDim
	{
		std::string name;
		uint32_t dim;

		namedDim(uint32_t dim, std::string name = "")
		{
			this->name = name;
			this->dim = dim;
		}
	};

	template<typename T>
	class tensor
	{
	private:
		std::vector<namedDim> dimension;
		cudabasic::gpuArray<T> arr;
		std::vector<const tensor<T>*> connections;

		void addConnection(const tensor<T>* newConnection)
		{
			connections.push_back(newConnection);
		}

		static bool canMatrixMultiply(const tensor<T>& a, const tensor<T>& b)
		{
			auto& aDims = a.getDimensions();
			auto& bDims = b.getDimensions();

			if (aDims.size() != 2 || bDims.size() != 2)
			{
				return false;
			}

			if (aDims[1].dim != bDims[0].dim)
			{
				return false;
			}

			return true;
		}

		static tensor<T>* createTensorWithMatrixMultiplyDims(const tensor<T>& a, const tensor<T>& b)
		{
			auto& aDims = a.getDimensions();
			auto& bDims = b.getDimensions();

			std::vector<uint32_t> new_dim;
			std::vector<std::string> new_name;

			new_dim.push_back(aDims[0].dim);
			new_dim.push_back(bDims[1].dim);

			for (size_t i = 0; i < aDims.size(); i++)
			{
				new_name.push_back(aDims.front().name != "" ? aDims[i].name : bDims[i].name);
			}

			return new tensor<T>(new_dim, new_name);
		}

	public:
		tensor(std::vector<uint32_t> dims) : tensor(dims, std::vector<std::string>(dims.size()))
		{ }
		tensor(std::vector<uint32_t> dims, std::vector<T> values) : tensor(dims, std::vector<std::string>(dims.size()), values)
		{ }
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names) : tensor(dimensions, names, std::vector<T>())
		{ }		
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values) : arr(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<uint32_t>()))
		{
			if (dimensions.size() == 0)
			{
				throw std::exception("Cannot make tensor with 0 dimensions.");
			}

			if (dimensions.size() != names.size())
			{
				throw std::exception("Number of dimensions and dimension names do not match.");
			}

			if (std::any_of(dimensions.begin(), dimensions.end(), [](auto& dim) { return dim == 0; }))
			{
				throw std::exception("Dimensions with size 0 are not allowed in a tensor.");
			}

			if (values.size() > arr.size())
			{
				throw std::exception("Initializtion vector contain too many values.");
			}

			for (size_t i = 0; i < dimensions.size(); i++)
			{
				dimension.push_back(namedDim(dimensions[i], names[i]));
			}

			if (values.size() > 0)
			{
				arr.copyToGPU(values);
			}
		}

		void makeRandom(T min, T max)
		{
			//std::default_random_engine rngGen;
			//std::uniform_real_distribution<T> dist(min, max);

			//for (uint32_t i = 0; i < arr.size(); i++)
			//{
			//	this->arr[i] = dist(rngGen);
			//}
		}
		uint32_t elementCount() const
		{
			return arr.size();
		}
		const std::vector<namedDim>& getDimensions() const
		{
			return dimension;
		}
		cudabasic::span<T> getGPUArray() const
		{
			return arr.getGPUArray();
		}
		const cudabasic::span<T> getGPUArrayConst() const
		{
			return arr.getGPUArrayConst();
		}
		std::vector<T> getValuesOnCPU() const
		{
			return arr.copyToCPU();
		}
		void transpose();
		void permute();
		void view();
		void resize();
		const matrix<T> getMatrixConst() const
		{
			return matrix<T>(arr.getGPUArrayConst().begin(), dimension[0].dim, dimension[1].dim);
		}
		matrix<T> getMatrix() const
		{
			return matrix<T>(arr.getGPUArrayConst().begin(), dimension[0].dim, dimension[1].dim);
		}
		tensor<T>* matMul(const tensor<T>& right) const
		{
			if (!canMatrixMultiply(this, right))
			{
				throw std::exception("Left hand side tensor cannot matrix multiply with right hand side tensor.");
			}

			tensor<T>* child = createTensorWithMatrixMultiplyDims(this, right);

			// make kernel call
			tensorMatrixMultiply(this, right, *child);

			return child;
		}
	};
}