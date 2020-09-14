#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <numeric>
#include "span.h"
#include "gpuArray.h"
#include "matrix.h"

namespace dnnbasic
{
	struct namedDim
	{
		std::string name;
		uint32_t dim;

		namedDim(uint32_t dim, std::string name);
	};

	template<typename T>
	class tensor
	{
	private:
		std::vector<namedDim> dimension;
		cudabasic::gpuArray<T> arr;
		std::vector<const tensor<T>*> connections;

		void addConnection(const tensor<T>* newConnection);

	public:
		tensor(std::vector<uint32_t> dims);
		tensor(std::vector<uint32_t> dims, std::vector<T> values);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names);
		tensor(std::vector<uint32_t> dimensions, std::vector<std::string> names, std::vector<T> values);

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

		tensor<T>* matMul(const tensor<T>& right) const;
		tensor<T>* permute(std::vector<uint32_t> dims) const;

	};
}