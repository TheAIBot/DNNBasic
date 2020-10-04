#pragma once
#include <cstdint>
#include <string>
#include <numeric>
#include <vector>
#include <memory>
#include "gpuArray.h"
#include "tensor_node.h"
#include "optional.h"
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
	class tensorData
	{
	private:
		std::vector<namedDim> dimension;
		cudabasic::gpuArray<T> arr;
		optional<std::shared_ptr<tensorNode<T>>> tensorOp;

	public:
		tensorData(std::vector<uint32_t> dimensions);


		matrix<T> getMatrix() const;
		matrix<T> getMatrixWith1Width() const;
		matrix<T> getMatrixWith1Height() const;


		friend class tensor<T>;
	};
}