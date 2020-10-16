#include <string>
#include <stdexcept>
#include <vector>
#include "tensor_permute_kernel.cuh"
#include "tensor.h"
#include "tensor_node_no_grad.h"
#include "auto_graph.h"

namespace dnnbasic
{
	template<typename T>
	static tensor<T> createTensorWithPermutedDims(const tensor<T>& a, std::vector<uint32_t> dims)
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			new_dim.push_back(aDims[dims[i]].dim);
			new_name.push_back(aDims[dims[i]].name);
		}

		return tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T> tensor<T>::permute(std::initializer_list<uint32_t> dims) const
	{
		return permute(std::vector<uint32_t>(dims));
	}

	template<typename T>
	tensor<T> tensor<T>::permute(std::vector<uint32_t> dims) const
	{
		if (dims.size() != this->data->dimension.size())
		{
			throw std::runtime_error("cannot perform permutation due to incorrect number of permute dimensions.");
		}
		for (uint32_t i = 0; i < this->data->dimension.size(); i++)
		{
			if (dims[i] >= this->data->dimension.size())
			{
				throw std::runtime_error("permute dimensions indicies cannot be higher than tensor dimnesion count.");
			}
		}

		tensor<T> child = createTensorWithPermutedDims(*this, dims);
		autoGraph::handleMakeGraph(child, std::function<tensorNode<T>*()>([&]() {return new tensorNodeNoGrad<T>({ *this }); }));

		// make kernel call
		tensorPermute(*this, child, dims);

		return child;
	}

	template<typename T>
	tensor<T> tensor<T>::permute(std::initializer_list<std::string> dimNames) const
	{
		return permute(std::vector<std::string>(dimNames));
	}

	template<typename T>
	tensor<T> tensor<T>::permute(std::vector<std::string> dimNames) const
	{
		if (dimNames.size() != this->getDimensions().size())
		{
			throw std::runtime_error("cannot perform permutation due to incorrect number of permute dimensions.");
		}

		std::vector<uint32_t> namedDimIndices;

		auto& tensorDims = this->getDimensions();
		for (size_t z = 0; z < dimNames.size(); z++)
		{
			const std::string& dimName = dimNames[z];
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
				throw std::runtime_error("Tensor does not contain a dimension with the name: ");
			}
		}
		
		return permute(namedDimIndices);
	}
}