#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <assert.h>
#include "optional.h"
#include "span.h"
#include "gpuArray.h"
#include "matrix.h"
#include "FBPropagation.h"

namespace dnnbasic
{
	struct namedDim
	{
		std::string name;
		uint32_t dim;

		namedDim(uint32_t dim, std::string name);
	};

	template<typename T>
	class tensorNode : public fbpropagation<T>
	{
	private:
		tensor<T> forward(const tensor<T>& x) const override
		{
			throw new std::runtime_error("Wait how you do that?");
		}

	};


	template<typename T>
	class tensorData
	{
	public:
		std::vector<namedDim> dimension;
		cudabasic::gpuArray<T> arr;
		optional<std::shared_ptr<tensorNode<T>>> tensorOp;

		tensorData(std::vector<uint32_t> dimensions) : arr(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<uint32_t>()))
		{

		}
	};

	template<typename T>
	class tensor
	{
	private:
		std::shared_ptr<tensorData<T>> data;

	public:
		static constexpr uint32_t MAX_DIMENSION_COUNT = 10;

		tensor(std::vector<uint32_t> dims);
		tensor(std::vector<uint32_t> dims, std::vector<T> values);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values);

		void setNode(tensorNode<T>* inNode);
		optional<std::shared_ptr<tensorNode<T>>> getNode();

		void makeRandom(T min, T max);
		uint32_t elementCount() const;
		const std::vector<namedDim>& getDimensions() const;
		cudabasic::span<T> getGPUArray() const;
		const cudabasic::span<T> getGPUArrayConst() const;
		std::vector<T> getValuesOnCPU() const;
		//void transpose();
		//void permute();
		//void view();
		//void resize();
		matrix<T> getMatrix() const;
		matrix<T> getMatrixWith1Width() const;
		matrix<T> getMatrixWith1Height() const;

		tensor<T> matMul(const tensor<T>& right) const;
		tensor<T> permute(std::initializer_list<uint32_t> dims) const;
		tensor<T> permute(std::vector<uint32_t> dims) const;
		tensor<T> permute(std::initializer_list<std::string> dims) const;
		tensor<T> permute(std::vector<std::string> dims) const;
	};

	template<typename T> bool operator==(const tensor<T>& left, const tensor<T>& right);
	template<typename T> bool operator!=(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator*(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator*(const tensor<T>& left, const T& right);
	template<typename T> tensor<T> operator*(const T& left, const tensor<T>& right);
	template<typename T> tensor<T> operator+(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator+(const tensor<T>& left, const T& right);
	template<typename T> tensor<T> operator+(const T& left, const tensor<T>& right);
	template<typename T> tensor<T> operator-(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator-(const tensor<T>& left, const T& right);
	template<typename T> tensor<T> operator-(const T& left, const tensor<T>& right);
	template<typename T> tensor<T> operator-(const tensor<T>& left);

}