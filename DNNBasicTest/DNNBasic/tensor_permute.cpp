#include "tensor_def.h"
#include "tensor_matrix_kernels.cuh"

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
}