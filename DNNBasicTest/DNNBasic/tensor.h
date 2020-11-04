#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include "optional.h"
#include "span.h"
#include "matrix.h"
#include "tensor_data.h"

namespace dnnbasic
{
	template<typename T>
	class tensor
	{
	private:
		std::shared_ptr<tensorData<T>> data;

	public:
		static constexpr uint32_t MAX_DIMENSION_COUNT = 10;

		static tensor<T> random(std::initializer_list<uint32_t> dims);
		static tensor<T> random(std::vector<uint32_t> dims);

		tensor(std::vector<uint32_t> dims);
		tensor(std::vector<uint32_t> dims, std::vector<T> values);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::shared_ptr<cudabasic::gpuArray<T>> tData);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values, std::shared_ptr<tensorData<T>> tData);

		void setNode(tensorNode<T>* inNode);
		optional<std::shared_ptr<tensorNode<T>>> getNode();

		uint32_t elementCount() const;
		const std::vector<namedDim>& getDimensions() const;
		cudabasic::span<T> getGPUArray() const;
		const cudabasic::span<T> getGPUArrayConst() const;
		std::vector<T> getValuesOnCPU() const;

		bool hasDimension(const std::string& dimName) const;
		uint32_t getDimensionIndex(const std::string& dimName) const;
		uint32_t getDimension(const uint32_t dimIdx) const;
		uint32_t getDimension(const std::string& dimName) const;

		void copyTo(const tensor<T>& other) const;

		tensor<T> matMul(const tensor<T>& right) const;

		tensor<T> permute(const std::initializer_list<namedDim>& dims) const;
		tensor<T> permute(const std::vector<namedDim>& dims) const;

		tensor<T> transpose(const uint32_t dim0, const uint32_t dim1) const;
		//tensor<T> transpose(const std::string& t1, const std::string& t2);

		template<typename U>
		tensor<U> cast() const;

		static tensor<T> exp(const tensor<T>& input);
		static tensor<T> log(const tensor<T>& input);

		tensor<T> max(const uint32_t maxDim) const;
		tensor<T> max(const std::string maxDim) const;

		tensor<T> sum(const uint32_t sumDim) const;
		tensor<T> sum(const std::string sumDim) const;

		tensor<T> reshape(const std::initializer_list<namedDim>& dims) const;
		tensor<T> reshape(const std::vector<namedDim>& dims) const;

		template<typename ...Ts> tensor<T> permute(const Ts& ... args) const 
		{ 
			return permute({ namedDim(args)... }); 
		}
		template<typename ...Ts> tensor<T> reshape(const Ts& ... args) const 
		{ 
			return reshape({ namedDim(args)... }); 
		}
	};

	template<typename T> bool operator==(const tensor<T>& left, const tensor<T>& right);
	template<typename T> bool operator!=(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator*(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator*(const tensor<T>& left, const T& right);
	template<typename T> tensor<T> operator*(const T& left, const tensor<T>& right);
	template<typename T> tensor<T> operator/(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator/(const tensor<T>& left, const T& right);
	template<typename T> tensor<T> operator+(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator+(const tensor<T>& left, const T& right);
	template<typename T> tensor<T> operator+(const T& left, const tensor<T>& right);
	template<typename T> tensor<T> operator-(const tensor<T>& left, const tensor<T>& right);
	template<typename T> tensor<T> operator-(const tensor<T>& left, const T& right);
	template<typename T> tensor<T> operator-(const T& left, const tensor<T>& right);
	template<typename T> tensor<T> operator-(const tensor<T>& left);
}