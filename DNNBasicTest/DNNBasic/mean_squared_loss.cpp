#include "mean_squared_loss.h"
#include "auto_graph.h"
#include <stdexcept>
#include <vector>

namespace dnnbasic::loss
{
	template<typename T>
	lossData<T>::lossData(tensor<T> gradient, T error, std::shared_ptr<tensorNode<T>> leafNode) :
		gradient(gradient),
		error(error),
		leafNode(leafNode) { }


	template<typename T>
	void lossData<T>::backward(optimizer::optimizer* opti)
	{
		this->leafNode->backward(this->gradient, opti);
	}


	template<typename T>
	lossData<T> meanSquaredLoss(tensor<T> expected, tensor<T> actual) 
	{
		if (!actual.getNode().has_value())
		{
			throw std::runtime_error("Can't make loss when argument actual is not part of a graph.");
		}

		autoGraph::scopeLevelDisableAutoGraph k;

		tensor<T> gradient = actual - expected;
		tensor<T> error = 0.5f * (gradient * gradient);

		std::vector<T> errorValues = error.getValuesOnCPU();
		T errorSum = std::accumulate(errorValues.begin(), errorValues.end(), (T)0);
		T meanError = errorSum / errorValues.size();


		return lossData(gradient, meanError, actual.getNode().value());
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
	//template lossData<double> meanSquaredLoss(tensor<double> expected, tensor<double> actual);

	template class lossData<bool>;
	template class lossData<uint8_t>;
	template class lossData<uint16_t>;
	template class lossData<uint32_t>;
	template class lossData<uint64_t>;
	template class lossData<int8_t>;
	template class lossData<int16_t>;
	template class lossData<int32_t>;
	template class lossData<int64_t>;
	template class lossData<float>;
	template class lossData<double>;
}
