#include <stdexcept>
#include <vector>
#include "cross_entropy_loss.h"
#include "auto_graph.h"
#include "activation_function.h"

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
		tensor<T> actualExp = dnnbasic::tensor<float>::exp(actual - actual.max(1));
		tensor<T> actualSumEXP = actualExp.sum(1);
		tensor<T> actualLogSumExp = dnnbasic::tensor<float>::log(actualSumEXP);
		tensor<T> minusLogSoftmax = (actualLogSumExp.reshape(actualLogSumExp.getDimension(0),1) - actual);

		tensor<T> error = (minusLogSoftmax * expected).sum(1);

		tensor<T> gradient = -dnnbasic::tensor<float>::exp(minusLogSoftmax) - expected;

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
