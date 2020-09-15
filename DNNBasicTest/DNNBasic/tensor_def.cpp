#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <numeric>
#include "span.h"
#include "gpuArray.h"
#include "matrix.h"
#include "tensor_def.h"

#include "tensor_permute.cpp"
#include "tensor_matrixMultiply.cpp"
#include "tensor_basic_math_operators.cpp"

namespace dnnbasic
{
	namedDim::namedDim(uint32_t dim, std::string name = "")
	{
		this->name = name;
		this->dim = dim;
	}

	template<typename T>
	void tensor<T>::addConnection(const tensor<T>* newConnection)
	{
		connections.push_back(newConnection);
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
	tensor<T>::tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values) : arr(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<uint32_t>()))
	{
		if (dimensions.size() == 0)
		{
			throw std::exception("Cannot make tensor with 0 dimensions.");
		}

		if (dimensions.size() != names.size())
		{
			throw std::exception("Number of dimensions and dimension names do not match.");
		}

		if (std::any_of(dimensions.begin(), dimensions.end(), [](auto& dim) { return dim == 0; }))
		{
			throw std::exception("Dimensions with size 0 are not allowed in a tensor.");
		}

		if (values.size() > arr.size())
		{
			throw std::exception("Initializtion vector contain too many values.");
		}

		for (size_t i = 0; i < dimensions.size(); i++)
		{
			dimension.push_back(namedDim(dimensions[i], names[i]));
		}

		if (values.size() > 0)
		{
			arr.copyToGPU(values);
		}
	}

	template<typename T>
	void tensor<T>::makeRandom(T min, T max)
	{
		//std::default_random_engine rngGen;
		//std::uniform_real_distribution<T> dist(min, max);

		//for (uint32_t i = 0; i < arr.size(); i++)
		//{
		//	this->arr[i] = dist(rngGen);
		//}
	}
	template<typename T>
	uint32_t tensor<T>::elementCount() const
	{
		return arr.size();
	}
	template<typename T>
	const std::vector<namedDim>& tensor<T>::getDimensions() const
	{
		return dimension;
	}
	template<typename T>
	cudabasic::span<T> tensor<T>::getGPUArray() const
	{
		return arr.getGPUArray();
	}
	template<typename T>
	const cudabasic::span<T> tensor<T>::getGPUArrayConst() const
	{
		return arr.getGPUArrayConst();
	}
	template<typename T>
	std::vector<T> tensor<T>::getValuesOnCPU() const
	{
		return arr.copyToCPU();
	}
	//void transpose();
	//void permute();
	//void view();
	//void resize();
	template<typename T>
	matrix<T> tensor<T>::getMatrix() const
	{
		return matrix<T>(arr.getGPUArrayConst().begin(), dimension[1].dim, dimension[0].dim);
	}
	template<typename T>
	matrix<T> tensor<T>::getMatrixWith1Width() const
	{
		return matrix<T>(arr.getGPUArrayConst().begin(), 1, dimension[0].dim);
	}
	template<typename T>
	matrix<T> tensor<T>::getMatrixWith1Height() const
	{
		return matrix<T>(arr.getGPUArrayConst().begin(), dimension[0].dim, 1);
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