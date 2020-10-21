#include <vector>
#include <string>
#include <stdexcept>
#include "tensor.h"
#include "tensor_node_no_grad.h"
#include "auto_graph.h"

namespace dnnbasic
{
	template<typename T>
	tensor<T> tensor<T>::reshape(std::initializer_list<namedDim> dims) const
	{
		return reshape(std::vector<namedDim>(dims));
	}

	template<typename T>
	tensor<T> tensor<T>::reshape(std::vector<namedDim> dims) const
	{
		uint32_t newTotalSize = 1;
		for (size_t i = 0; i < dims.size(); i++)
		{
			newTotalSize *= dims[i].hasName() ? this->getDimension(dims[i].name) : dims[i].dim;
		}

		if (newTotalSize != this->elementCount())
		{
			throw std::runtime_error("Reshaping has to result in a tensor with the same size.");
		}

		std::vector<uint32_t> newDims;
		std::vector<std::string> newDimNames;

		for (size_t i = 0; i < dims.size(); i++)
		{
			newDims.push_back(dims[i].hasName() ? this->getDimension(dims[i].name) : dims[i].dim);
			newDimNames.push_back(dims[i].hasName() ? dims[i].name : this->data->dimension[i].name);
		}

		tensor<T> reshaped(newDims, newDimNames);
		autoGraph::handleMakeGraph(reshaped, std::function<tensorNode<T>* ()>([&]() {return new tensorNodeNoGrad<T>({ *this }); }));

		this->copyTo(reshaped);

		return reshaped;
	}
}