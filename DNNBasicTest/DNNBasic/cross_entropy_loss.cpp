#include <stdexcept>
#include <vector>
#include "cross_entropy_loss.h"
#include "auto_graph.h"
#include "activation_function.h"
#include "tensor_activation_kernels.cuh"
#include "tensor_node_no_grad.h"

namespace dnnbasic::loss
{
	template<typename T>
	lossData<T> crossEntropyLoss(tensor<T> expected, tensor<T> actual, bool meanOverBatch, const uint32_t batchDim)
	{
		if (!actual.getNode().has_value())
		{
			throw std::runtime_error("Can't make loss when argument actual is not part of a graph.");
		}

		autoGraph::scopeLevelDisableAutoGraph k;

		// softmax Kernel
		auto castqz = actual + 0.0f;
		tensorReLU(actual, castqz);
		autoGraph::handleMakeGraph(castqz, std::function<tensorNode<T>* ()>([&]() { return new tensorNodeNoGrad<T>({ actual }); }));
		auto castz = castqz.cast<uint32_t>();

		auto maxVals = castz.max(1).cast<float>().reshape(actual.getDimension(0), 1);

		tensor<T> actualExp = dnnbasic::tensor<float>::exp(actual - maxVals);
		tensor<T> actualSumEXP = actualExp.sum(1);
		tensor<T> softmax = actualExp / (actualSumEXP.reshape(actualSumEXP.getDimension(0), 1));

		tensor<T> error = -expected * dnnbasic::tensor<float>::log(0.0000001f + softmax);
		error = error.sum(1);

		tensor<T> gradient = (softmax - expected);

		auto errorMethod = [](const tensor<T>& ten)
		{
			std::vector<T> errorValues = ten.getValuesOnCPU();
			T errorSum = std::accumulate(errorValues.begin(), errorValues.end(), (T)0);
			return errorSum / errorValues.size();
		};


		return lossData<T>(gradient, error, actual.getNode().value(), errorMethod);
	}

	template<typename T>
	lossData<T> crossEntropyLoss(tensor<T> expected, tensor<T> actual)
	{
		return crossEntropyLoss(expected, actual, false, 0);
	}

	template<typename T>
	lossData<T> crossEntropyLoss(tensor<T> expected, tensor<T> actual, const uint32_t batchDim)
	{
		return crossEntropyLoss(expected, actual, true, batchDim);
	}

	template lossData<float> crossEntropyLoss(tensor<float> expected, tensor<float> actual);
	template lossData<float> crossEntropyLoss(tensor<float> expected, tensor<float> actual, const uint32_t batchDim);
	template lossData<float> crossEntropyLoss(tensor<float> expected, tensor<float> actual, bool meanOverBatch, const uint32_t batchDim);

}
