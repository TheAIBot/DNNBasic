#include <stdexcept>
#include <vector>
#include "mean_squared_loss.h"
#include "auto_graph.h"
#include "activation_function.h"

namespace dnnbasic::loss
{
	
	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual, bool meanOverBatch, const uint32_t batchDim)
	{
		if (!actual.getNode().has_value())
		{
			throw std::runtime_error("Can't make loss when argument actual is not part of a graph.");
		}

		autoGraph::scopeLevelDisableAutoGraph k;

		tensor<T> gradient = actual - expected;
		//if (meanOverBatch)
		//{
		//	gradient = gradient.sum(batchDim) / (T)gradient.getDimensions()[batchDim].dim;
		//}
		tensor<T> error = 0.5f * (gradient * gradient);

		auto errorMethod = [](const tensor<T>& ten)
		{
			std::vector<T> errorValues = ten.getValuesOnCPU();
			T errorSum = std::accumulate(errorValues.begin(), errorValues.end(), (T)0);
			return errorSum / errorValues.size();
		};


		return lossData<T>(gradient, error, actual.getNode().value(), errorMethod);
	}

	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual) 
	{
		return meanSquaredLoss(expected, actual, false, 0);
	}

	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual, const uint32_t batchDim)
	{
		return meanSquaredLoss(expected, actual, true, batchDim);
	}

	//template lossData<bool> meanSquaredLoss(tensor<bool> expected, tensor<bool> actual);
	//template lossData<uint8_t> meanSquaredLoss(tensor<uint8_t> expected, tensor<uint8_t> actual);
	//template lossData<uint16_t> meanSquaredLoss(tensor<uint16_t> expected, tensor<uint16_t> actual);
	//template lossData<uint32_t> meanSquaredLoss(tensor<uint32_t> expected, tensor<uint32_t> actual);
	//template lossData<uint64_t> meanSquaredLoss(tensor<uint64_t> expected, tensor<uint64_t> actual);
	//template lossData<int8_t> meanSquaredLoss(tensor<int8_t> expected, tensor<int8_t> actual);
	//template lossData<int16_t> meanSquaredLoss(tensor<int16_t> expected, tensor<int16_t> actual);
	//template lossData<int32_t> meanSquaredLoss(tensor<int32_t> expected, tensor<int32_t> actual);
	//template lossData<int64_t> meanSquaredLoss(tensor<int64_t> expected, tensor<int64_t> actual);
	template lossData<float> meanSquaredLoss(tensor<float> expected, tensor<float> actual);
	template lossData<float> meanSquaredLoss(tensor<float> expected, tensor<float> actual, const uint32_t batchDim);
	template lossData<float> meanSquaredLoss(tensor<float> expected, tensor<float> actual, bool meanOverBatch, const uint32_t batchDim);
	//template lossData<double> meanSquaredLoss(tensor<double> expected, tensor<double> actual);

}
