#include <string>
#include <stdexcept>
#include "tensor_max_kernel.cuh"
#include "tensor.h"
#include "tensor_node_no_grad.h"
#include "auto_graph.h"

namespace dnnbasic
{
	template<typename T>
	static tensor<T> createTensorWithSameDimsButWithoutMaxDim(const tensor<T>& a, const uint32_t maxDim)
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			if (i == maxDim)
			{
				continue;
			}
			new_dim.push_back(aDims[i].dim);
			new_name.push_back(aDims[i].name);
		}

		return tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T> tensor<T>::max(const uint32_t maxDim) const
	{
		if (maxDim >= this->getDimensions().size())
		{
			throw std::runtime_error("Sum dimension index cannot be higher than tensor dimnesion count.");
		}

		tensor<T> child = createTensorWithSameDimsButWithoutMaxDim(*this, maxDim);
		autoGraph::handleMakeGraph(child, std::function<tensorNode<T>* ()>([&]() {return new tensorNodeNoGrad<T>({ *this }); }));

		tensorMax(*this, child, maxDim);

		return child;
	}

	template<typename T>
	tensor<T> tensor<T>::max(const std::string maxDim) const
	{
		uint32_t dim = 0;
		bool foundDim = false;
		for (uint32_t i = 0; i < this->getDimensions().size(); i++)
		{
			if (this->getDimensions()[i].name == maxDim)
			{
				dim = i;
				foundDim = true;
				break;
			}
		}

		if (!foundDim)
		{
			throw std::runtime_error("Tensor does not contain a dimension withthat name.");
		}

		return max(dim);
	}
}