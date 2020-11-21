#include "relu.h"
#include "tensor_node_linear.h"
#include "auto_graph.h"
#include "tensor_activation_kernels.cuh"
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

		//tensor<T> relu<T>::derivative(const tensor<T>& input)
		template<typename T>
		tensor<T> relu<T>::derivative(const tensor<T>& derivative_activation_function, const tensor<T>& affine_input)
		{

			auto& inputDims = affine_input.getDimensions();

			std::vector<uint32_t> new_dim;
			std::vector<std::string> new_name;
			for (size_t i = 0; i < inputDims.size(); i++)
			{
				new_dim.push_back(inputDims[i].dim);
				new_name.push_back(inputDims[i].name);
			}

			tensor<T> output = tensor<T>(new_dim, new_name);

			// make new node
			autoGraph::handleMakeGraph(output, std::function<tensorNode<T>* ()>([&]() { return new tensorNodeNoGrad<T>({ derivative_activation_function, affine_input }); }));

			tensorReLUDerivative(derivative_activation_function, affine_input, output);

			return output;
		}

		template class relu<uint8_t>;
		template class relu<uint16_t>;
		template class relu<uint32_t>;
		template class relu<uint64_t>;
		template class relu<int8_t>;
		template class relu<int16_t>;
		template class relu<int32_t>;
		template class relu<int64_t>;
		template class relu<float>;
		template class relu<double>;
	}
}