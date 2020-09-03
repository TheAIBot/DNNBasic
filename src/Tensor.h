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

		tensor<T>* operator*(tensor<T>& input) 
		{
			if (input.dimension.size() != this->dimension.size())
			{
				throw std::exception("Dimension of input doesn't match dimension of tensor.");
			}

			std::vector<uint32_t> new_dim;
			std::vector<std::string> new_name;
			for (size_t i = 0; i < input.dimension.size(); i++)
			{
				if (this->dimension[i].dim != input.dimension[i].dim)
				{
					throw std::exception("Input output dimension mismatch.");
				}
				new_dim.push_back(this->dimension[i].dim);
				new_name.push_back(this->dimension[i].name);
			}

			tensor<T>* child = new tensor<T>(new_dim, new_name);
			child->addConnection(this);
			child->addConnection(&input);

			// make kernel call

			return child;

		}

		T& operator[](const uint32_t i)
		{
			return arr[i];
		}

		T& operator[](const uint32_t i) const
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
	
}