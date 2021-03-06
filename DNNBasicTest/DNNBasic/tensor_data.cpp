#include "tensor_data.h"

namespace dnnbasic
{
	namedDim::namedDim(uint32_t dim) : dim(dim)
	{
	}
	namedDim::namedDim(std::string name) : dim(0), name(name)
	{
	}
	namedDim::namedDim(uint32_t dim, std::string name) : dim(dim), name(name)
	{
	}

	bool namedDim::hasName() const
	{
		return !this->name.empty();
	}

	template<typename T>
	tensorData<T>::tensorData(std::vector<uint32_t> dimensions) : arr(std::make_shared<cudabasic::gpuArray<T>>(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<uint32_t>())))
	{ }
	template<typename T>
	tensorData<T>::tensorData(std::vector<uint32_t> dimensions, std::shared_ptr<cudabasic::gpuArray<T>> gpuArr) : arr(gpuArr)
	{ }

	template<typename T>
	matrix<T> tensorData<T>::getMatrix() const
	{
		return matrix<T>(this->arr->getGPUArrayConst().begin(), this->dimension[1].dim, this->dimension[0].dim);
	}
	template<typename T>
	matrix<T> tensorData<T>::getMatrixWith1Width() const
	{
		return matrix<T>(this->arr->getGPUArrayConst().begin(), 1, this->dimension[0].dim);
	}
	template<typename T>
	matrix<T> tensorData<T>::getMatrixWith1Height() const
	{
		return matrix<T>(this->arr->getGPUArrayConst().begin(), this->dimension[0].dim, 1);
	}

	template class tensorData<bool>;
	template class tensorData<uint8_t>;
	template class tensorData<uint16_t>;
	template class tensorData<uint32_t>;
	template class tensorData<uint64_t>;
	template class tensorData<int8_t>;
	template class tensorData<int16_t>;
	template class tensorData<int32_t>;
	template class tensorData<int64_t>;
	template class tensorData<float>;
	template class tensorData<double>;
}