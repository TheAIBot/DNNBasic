#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <numeric>
#include "span.h"
#include "gpuArray.h"

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
		std::vector<tensor<T>*> connections;

		void addConnection(tensor<T>* newConnection)
		{
			connections.push_back(newConnection);
		}

	public:
		tensor(std::vector<uint32_t> dims) : tensor(dims, std::vector<std::string>(dims.size()))
		{ }
		tensor(std::vector<uint32_t> dims, std::vector<T> values) : tensor(dims, std::vector<std::string>(dims.size()), values)
		{ }
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names) : tensor(dimensions, names, std::vector<T>())
		{ }		
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values) : arr(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<T>()))
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

		template<typename U> friend bool operator==(tensor<U>&, tensor<U>&);
		template<typename U> friend bool operator!=(tensor<U>&, tensor<U>&);
		template<typename U> friend bool operator==(const tensor<U>&, const tensor<U>&);
		template<typename U> friend bool operator!=(const tensor<U>&, const tensor<U>&);

		//template<typename U> friend tensor<U>* operator+(tensor<U>&, tensor<U>&);
		//template<typename U> friend tensor<U>* operator+(T&, tensor<U>&);
		//template<typename U> friend tensor<U>* operator+(tensor<U>&, T&);

		//template<typename U> friend tensor<U>* operator-(tensor<U>&, tensor<U>&);
		//template<typename U> friend tensor<U>* operator-(T&, tensor<U>&);
		//template<typename U> friend tensor<U>* operator-(tensor<U>&, T&);

		template<typename U> friend tensor<U>* operator*(tensor<U>&, tensor<U>&);
		//template<typename U> friend tensor<U>* operator*(T&, tensor<U>&);
		//template<typename U> friend tensor<U>* operator*(tensor<U>&, T&);

		//T& operator[](const uint32_t i)
		//{
		//	return arr[i];
		//}

		//T operator[](const uint32_t i) const
		//{
		//	return arr[i];
		//}

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
	};
}