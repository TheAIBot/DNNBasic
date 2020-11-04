#include <string>
#include <stdexcept>
#include <vector>
#include "tensor_log_kernel.cuh"
#include "tensor.h"
#include "tensor_node_no_grad.h"
#include "auto_graph.h"

namespace dnnbasic
{
	template<typename T>
	tensor<T> tensor<T>::log(const tensor<T>& a)
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			new_dim.push_back(aDims[i].dim);
			new_name.push_back(aDims[i].name);
		}

		tensor<T> child = tensor<T>(new_dim, new_name);

		autoGraph::handleMakeGraph(child, std::function<tensorNode<T>* ()>([&]() {return new tensorNodeNoGrad<T>({ a }); }));

		tensorLog(a, child);

		return child;
	}

}