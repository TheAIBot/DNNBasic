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
	tensor<T> tensor<T>::permute(const std::initializer_list<namedDim>& dims) const
	{
		return permute(std::vector<namedDim>(dims));
	}

	template<typename T>
	tensor<T> tensor<T>::permute(const std::vector<namedDim>& dims) const
	{
		if (dims.size() != this->data->dimension.size())
		{
			throw std::runtime_error("Cannot perform permutation due to incorrect number of permute dimensions.");
		}

		std::vector<uint32_t> actualDims;
		for (uint32_t i = 0; i < this->data->dimension.size(); i++)
		{
			if (dims[i].hasName() && !this->hasDimension(dims[i].name))
			{
				throw std::runtime_error("Tensor does not contain a dimension with the name: ");
			}
			if (dims[i].dim >= this->data->dimension.size())
			{
				throw std::runtime_error("Permute dimensions indicies cannot be higher than tensor dimnesion count.");
			}

			actualDims.push_back(dims[i].hasName() ? this->getDimensionIndex(dims[i].name) : dims[i].dim);
		}

		tensor<T> child = createTensorWithPermutedDims(*this, actualDims);
		autoGraph::handleMakeGraph(child, std::function<tensorNode<T>*()>([&]() {return new tensorNodeNoGrad<T>({ *this }); }));

		// make kernel call
		tensorPermute(*this, child, actualDims);

		return child;
	}

	template<typename T>
	tensor<T> tensor<T>::transpose(const uint32_t dim0, const uint32_t dim1) const
	{
		if (dim0 >= this->getDimensions().size())
		{
			throw std::runtime_error("Transpose dimension indicie cannot be higher than tensor dimnesion count.");
		}
		if (dim1 >= this->getDimensions().size())
		{
			throw std::runtime_error("Transpose dimension indicie cannot be higher than tensor dimnesion count.");
		}

		std::vector<namedDim> dims = this->getDimensions();
		
		for (size_t i = 0; i < dims.size(); i++)
		{
			dims[i].dim = i;
		}

		const namedDim tmp = dims[dim0];
		dims[dim0] = dims[dim1];
		dims[dim1] = tmp;

		return this->permute(dims);
	}
}