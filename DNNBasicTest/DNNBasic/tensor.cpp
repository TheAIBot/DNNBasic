#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "span.h"
#include "gpuArray.h"
#include "matrix.h"
#include "tensor.h"
#include "random.h"
#include "optional.h"
#include "cuda_settings.h"
#include "auto_graph.h"

#include "tensor_exp.cpp"
#include "tensor_log.cpp"
#include "tensor_sum.cpp"
#include "tensor_cast.cpp"
#include "tensor_permute.cpp"
#include "tensor_reshape.cpp"
#include "tensor_matrixMultiply.cpp"
#include "tensor_basic_math_operators.cpp"

namespace dnnbasic
{
	template<typename T>
	tensor<T> tensor<T>::random(std::initializer_list<uint32_t> dims)
	{
		std::vector<uint32_t> vDims = dims;
		return random(vDims);
	}
	template<typename T>
	tensor<T> tensor<T>::random(std::vector<uint32_t> dims)
	{
		const uint32_t sum = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>());
		if constexpr (std::is_unsigned<T>::value)
		{
			return tensor<T>(dims, random::getRandomNumbers<T>(sum, (T)0, (T)2));
		}
		else
		{
			return tensor<T>(dims, random::getRandomNumbers<T>(sum, (T)-1, (T)1));
		}
	}

	template<typename T>
	tensor<T>::tensor(std::vector<uint32_t> dims) : tensor(dims, std::vector<std::string>(dims.size()))
	{ }
	template<typename T>
	tensor<T>::tensor(std::vector<uint32_t> dims, std::vector<T> values) : tensor(dims, std::vector<std::string>(dims.size()), values)
	{ }
	template<typename T>
	tensor<T>::tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names) : tensor(dimensions, names, std::vector<T>())
	{ }
	template<typename T>
	tensor<T>::tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values)
	{
		this->data = std::make_shared<tensorData <T>>(dimensions);
		if (dimensions.size() == 0)
		{
			throw std::runtime_error("Cannot make tensor with 0 dimensions.");
		}

		if (dimensions.size() > tensor<T>::MAX_DIMENSION_COUNT)
		{
			throw std::runtime_error("A tensor can not have more than 10 dimensions.");
		}

		if (dimensions.size() != names.size())
		{
			throw std::runtime_error("Number of dimensions and dimension names do not match.");
		}

		if (std::any_of(dimensions.begin(), dimensions.end(), [](auto& dim) { return dim == 0; }))
		{
			throw std::runtime_error("Dimensions with size 0 are not allowed in a tensor.");
		}

		if (values.size() > this->data->arr.size())
		{
			throw std::runtime_error("Initializtion vector contain too many values.");
		}

		for (size_t i = 0; i < dimensions.size(); i++)
		{
			this->data->dimension.push_back(namedDim(dimensions[i], names[i]));
		}

		if (values.size() > 0)
		{
			this->data->arr.copyToGPU(values);
		}
	}

	template<typename T>
	void tensor<T>::setNode(tensorNode<T>* inNode)
	{
		this->data->tensorOp = optional(std::shared_ptr<tensorNode<T>>(inNode));
	}

	template<typename T>
	optional<std::shared_ptr<tensorNode<T>>> tensor<T>::getNode()
	{
		return this->data->tensorOp;
	}

	template<typename T>
	uint32_t tensor<T>::elementCount() const
	{
		return this->data->arr.size();
	}
	template<typename T>
	const std::vector<namedDim>& tensor<T>::getDimensions() const
	{
		return this->data->dimension;
	}
	template<typename T>
	cudabasic::span<T> tensor<T>::getGPUArray() const
	{
		return this->data->arr.getGPUArray();
	}
	template<typename T>
	const cudabasic::span<T> tensor<T>::getGPUArrayConst() const
	{
		return this->data->arr.getGPUArrayConst();
	}
	template<typename T>
	std::vector<T> tensor<T>::getValuesOnCPU() const
	{
		return this->data->arr.copyToCPU();
	}
	template<typename T>
	bool tensor<T>::hasDimension(const std::string& dimName) const
	{
		for (size_t i = 0; i < this->data->dimension.size(); i++)
		{
			if (this->data->dimension[i].name == dimName)
			{
				return true;
			}
		}

		return false;
	}
	template<typename T>
	uint32_t tensor<T>::getDimensionIndex(const std::string& dimName) const
	{
		for (uint32_t i = 0; i < this->data->dimension.size(); i++)
		{
			if (this->data->dimension[i].name == dimName)
			{
				return i;
			}
		}

		throw std::runtime_error("Tensor does not have a dimension with that name.");
	}
	template<typename T>
	uint32_t tensor<T>::getDimension(const uint32_t dimIdx) const
	{
		if (dimIdx >= this->data->dimension.size())
		{
			throw std::runtime_error("Tried to access a dimension that this tensor doesn't have.");
		}

		return this->data->dimension[dimIdx].dim;
	}
	template<typename T>
	uint32_t tensor<T>::getDimension(const std::string& dimName) const
	{
		for (size_t i = 0; i < this->data->dimension.size(); i++)
		{
			if (this->data->dimension[i].name == dimName)
			{
				return this->data->dimension[i].dim;
			}
		}

		throw std::runtime_error("Tensor does not have a dimension with that name.");
	}

	template<typename T>
	void tensor<T>::copyTo(const tensor<T>& other) const
	{
		if (autoGraph::isRecordingGraph())
		{
			autoGraph::addMemcpyNode(this->data->arr.getGPUArray(), other.data->arr.getGPUArray());
		}
		else
		{
			this->data->arr.copyToGPUArray(other.data->arr, cuda::getDefaultStream());
		}
	}


	template class tensor<bool>;
	template class tensor<uint8_t>;
	template class tensor<uint16_t>;
	template class tensor<uint32_t>;
	template class tensor<uint64_t>;
	template class tensor<int8_t>;
	template class tensor<int16_t>;
	template class tensor<int32_t>;
	template class tensor<int64_t>;
	template class tensor<float>;
	template class tensor<double>;
}