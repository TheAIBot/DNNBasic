#include "relu.h"
#include "tensor_node_linear.h"
#include "auto_graph.h"
#include "tensor_activation_kernels.cuh"
#include "relu.h"
#include "tensor_node_activation.h"
#include "tensor_node_no_grad.h"

namespace dnnbasic
{
	namespace activations 
	{
		template<typename T>
		tensor<T> relu<T>::forward(const tensor<T>& input)
		{
			auto& inputDims = input.getDimensions();

			std::vector<uint32_t> new_dim;
			std::vector<std::string> new_name;
			for (size_t i = 0; i < inputDims.size(); i++)
			{
				new_dim.push_back(inputDims[i].dim);
				new_name.push_back(inputDims[i].name);
			}

			tensor<T> output = tensor<T>(new_dim, new_name);

			// make new node
			autoGraph::handleMakeGraph(output, std::function<tensorNode<T>* ()>([&]() { return new tensorNodeActivation<T>(input, this); }));

			tensorReLU(input, output);

			return output;
		}

		template<typename T>
		tensor<T> relu<T>::derivative(const tensor<T>& input)
		{

			auto& inputDims = input.getDimensions();

			std::vector<uint32_t> new_dim;
			std::vector<std::string> new_name;
			for (size_t i = 0; i < inputDims.size(); i++)
			{
				new_dim.push_back(inputDims[i].dim);
				new_name.push_back(inputDims[i].name);
			}

			tensor<T> output = tensor<T>(new_dim, new_name);

			// make new node
			autoGraph::handleMakeGraph(output, std::function<tensorNode<T>* ()>([&]() { return new tensorNodeNoGrad<T>({ input }); }));

			tensorReLU(input, output);

			return output;
		}
	}
}