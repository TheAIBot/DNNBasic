#include "mean_squared_loss.h"
#include "auto_graph.h"
#include <stdexcept>

namespace dnnbasic::loss
{
	template<typename T>
	lossData<T>::lossData(tensor<T> gradient, tensor<T> error, std::shared_ptr<tensorNode<T>> leafNode) :
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
		if (actual.getNode().hasValue())
		{
			throw std::runtime_error("Can't make loss when argument actual is not part of a graph.");
		}

		autoGraph::scopeLevelDisableAutoGraph k;

		tensor<T> gradient = actual - expected;
		tensor<T> error = 0.5f * (gradient * gradient);

		return lossData(gradient, error, actual.getNode().value());
	}
}
