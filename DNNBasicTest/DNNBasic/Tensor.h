#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>
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

		tensor<T> matMul(const tensor<T>& right) const;
		tensor<T> permute(std::initializer_list<uint32_t> dims) const;
		tensor<T> permute(std::vector<uint32_t> dims) const;
		tensor<T> permute(std::initializer_list<std::string> dims) const;
		tensor<T> permute(std::vector<std::string> dims) const;

		template<typename U>
		tensor<U> cast() const;
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