#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "span.h"
#include <random>

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
		cudabasic::span<T> arr;
		std::vector<tensor<T>*> connections;

		void addConnection(tensor<T>* newConnection) 
		{
			connections.push_back(newConnection);
		}

		static bool hasSameDimensions(tensor<T>& a, tensor<T>& b)
		{
			if (a.dimension.size() != b.dimension.size())
			{
				return false;
			}

			for (size_t i = 0; i < a.dimension.size(); i++)
			{
				if (a.dimension[i].dim != b.dimension[i].dim)
				{
					return false;
				}
			}

			return true;
		}

	public:
		tensor(std::vector<uint32_t> dim) : tensor(dim, std::vector<std::string>(dim.size()))
		{ }
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names)
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

			uint32_t length = 1;
			for (size_t i = 0; i < dimensions.size(); i++)
			{
				length *= dimensions[i];
				dimension.push_back(namedDim(dimensions[i], names[i]));
			}

			this->arr = cudabasic::span<T>(new T[length], length);
		}
		~tensor() noexcept(false) 
		{
			delete[] arr.begin();
		};

		//tensor<T>* operator*(tensor<T>& input) 
		//{
		//	if (!hasSameDimensions(*this, input))
		//	{
		//		throw std::exception("Dimensions of input tensor doesn't match dimensions of tensor.");
		//	}

		//	std::vector<uint32_t> new_dim;
		//	std::vector<std::string> new_name;
		//	for (size_t i = 0; i < dimension.size(); i++)
		//	{
		//		new_dim.push_back(dimension[i].dim);
		//		new_name.push_back(dimension.front().name != "" ? dimension[i].name : input.dimension[i].name);
		//	}

		//	tensor<T>* child = new tensor<T>(new_dim, new_name);
		//	child->addConnection(this);
		//	child->addConnection(&input);

		//	// make kernel call
		//	for (size_t i = 0; i < child->elementCount(); i++)
		//	{
		//		(*child)[i] = (*this)[i] * input[i];
		//	}

		//	return child;
		//}

		template<typename U> friend tensor<U>* operator*(tensor<U>&, tensor<U>&);

		T& operator[](const uint32_t i)
		{
			return arr[i];
		}

		const T& operator[](const uint32_t i) const
		{
			return arr[i];
		}

		void makeRandom(T min, T max) 
		{
			std::default_random_engine rngGen;
			std::uniform_real_distribution<T> dist(min, max);

			for (uint32_t i = 0; i < arr.size(); i++)
			{
				this->arr[i] = dist(rngGen);
			}
		}
		uint32_t elementCount() const
		{
			return arr.size();
		}
		void transpose();
		void permute();
		void view();
		void resize();
	};

	template<typename T>
	tensor<T>* operator*(tensor<T>& left, tensor<T>& right)
	{
		if (!tensor<T>::hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of input tensor doesn't match dimensions of tensor.");
		}

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < left.dimension.size(); i++)
		{
			new_dim.push_back(left.dimension[i].dim);
			new_name.push_back(left.dimension.front().name != "" ? left.dimension[i].name : right.dimension[i].name);
		}

		tensor<T>* child = new tensor<T>(new_dim, new_name);
		child->addConnection(&left);
		child->addConnection(&right);

		// make kernel call
		for (size_t i = 0; i < child->elementCount(); i++)
		{
			(*child)[i] = left[i] * right[i];
		}

		return child;
	}
	
}