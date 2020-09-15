#include <string>
#include "tensor_permute_kernel.cuh"
#include "tensor.h"

namespace dnnbasic
{
	template<typename T>
	static tensor<T>* createTensorWithPermutedDims(const tensor<T>& a, std::vector<uint32_t> dims)
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			new_dim.push_back(aDims[dims[i]].dim);
			new_name.push_back(aDims[dims[i]].name);
		}

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T>* tensor<T>::permute(std::initializer_list<uint32_t> dims) const
	{
		return permute(std::vector<uint32_t>(dims));
	}

	template<typename T>
	tensor<T>* tensor<T>::permute(std::vector<uint32_t> dims) const
	{
		if (dims.size() != dimension.size())
		{
			throw std::exception("cannot perform permutation due to incorrect number of permute dimensions.");
		}
		for (uint32_t i = 0; i < dimension.size(); i++)
		{
			if (dims[i] >= dimension.size())
			{
				throw std::exception("permute dimensions indicies cannot be higher than tensor dimnesion count.");
			}
		}

		tensor<T>* child = createTensorWithPermutedDims(*this, dims);

		// make kernel call
		tensorPermute(*this, *child, dims);

		return child;
	}

	template<typename T>
	tensor<T>* tensor<T>::permute(std::initializer_list<std::string> dimNames) const
	{
		return permute(std::vector<std::string>(dimNames));
	}

	template<typename T>
	tensor<T>* tensor<T>::permute(std::vector<std::string> dimNames) const
	{
		if (dimNames.size() != this->getDimensions().size())
		{
			throw std::exception("cannot perform permutation due to incorrect number of permute dimensions.");
		}

		std::vector<uint32_t> namedDimIndices;

		auto& tensorDims = this->getDimensions();
		for each (const std::string & dimName in dimNames)
		{
			bool foundDim = false;
			for (uint32_t i = 0; i < tensorDims.size(); i++)
			{
				if (tensorDims[i].name == dimName)
				{
					namedDimIndices.push_back(i);
					foundDim = true;
					break;
				}
			}

			if (!foundDim)
			{
				throw std::exception("Tensor does not contain a dimension with the name: ");
			}
		}

		return permute(namedDimIndices);
	}
}