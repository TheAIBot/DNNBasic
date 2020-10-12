#include <string>
#include <stdexcept>
#include "tensor_sum_kernel.cuh"
#include "tensor.h"
#include "tensor_node_no_grad.h"
#include "auto_graph.h"

namespace dnnbasic
{
	template<typename T>
	static tensor<T> createTensorWithSameDimsButWithoutSumDim(const tensor<T>& a, const uint32_t sumDim)
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			if (i == sumDim)
			{
				continue;
			}
			new_dim.push_back(aDims[i].dim);
			new_name.push_back(aDims[i].name);
		}

		return tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T> tensor<T>::sum(const uint32_t sumDim) const
	{
		if (sumDim >= this->getDimensions().size())
		{
			throw std::runtime_error("Sum dimension index cannot be higher than tensor dimnesion count.");
		}

		tensor<T> child = createTensorWithSameDimsButWithoutSumDim(*this, sumDim);
		autoGraph::handleMakeGraph(child, std::function<tensorNode<T>*()>([&]() {return new tensorNodeNoGrad<T>({ *this }); }));

		tensorSum(*this, child, sumDim);

		return child;
	}

	template<typename T>
	tensor<T> tensor<T>::sum(const std::string sumDim) const
	{
		uint32_t dim = 0;
		bool foundDim = false;
		for (size_t i = 0; i < this->getDimensions().size(); i++)
		{
			if (this->getDimensions()[i].name == sumDim)
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

		return sum(dim);
	}
}