#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "span.h"
#include "tensor_def.h"
#include "tensor_elementwise_kernels.cuh"

namespace dnnbasic
{
	template<typename T>
	static bool hasSameDimensions(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		if (aDims.size() != bDims.size())
		{
			return false;
		}

		for (size_t i = 0; i < aDims.size(); i++)
		{
			if (aDims[i].dim != bDims[i].dim)
			{
				return false;
			}
		}

		return true;
	}

	template<typename T>
	static tensor<T>* createTensorWithSameDims(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			new_dim.push_back(aDims[i].dim);
			new_name.push_back(aDims.front().name != "" ? aDims[i].name : bDims[i].name);
		}

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	static tensor<T>* createTensorWithSameDims(const tensor<T>& a) 
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			new_dim.push_back(aDims[i].dim);
			new_name.push_back(aDims[i].name);
		}

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	static tensor<T>* createTensorWithPermutedDims(const tensor<T>& a, std::vector<uint32_t> dims)
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			new_dim.push_back(aDims[dims[i]].dim);
			new_name.push_back(aDims[dims[i]].name);
		}

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	bool operator==(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		auto leftValues = left.getValuesOnCPU();
		auto rightValues = right.getValuesOnCPU();

		return leftValues == rightValues;
	}

	template<typename T>
	bool operator!=(const tensor<T>& left, const tensor<T>& right)
	{
		return !(left == right);
	}

	template<typename T>
	tensor<T>* operator*(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		tensor<T>* child = createTensorWithSameDims(left, right);

		// make kernel call
		tensorMultiply(left, right, *child);

		return child;
	}

	template<typename T>
	tensor<T>* operator*(const tensor<T>& left, const T& right)
	{
		return right * left;
	}

	template<typename T>
	tensor<T>* operator*(const T& left, const tensor<T>& right)
	{
		tensor<T>* child = createTensorWithSameDims(right);

		// make kernel call
		tensorMultiply(left, right, *child);

		return child;
	}

	template<typename T>
	tensor<T>* operator+(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		tensor<T>* child = createTensorWithSameDims(left, right);

		// make kernel call
		tensorAdd(left, right, *child);

		return child;
	}

	template<typename T>
	tensor<T>* operator+(const tensor<T>& left, const T& right)
	{
		return right + left;
	}

	template<typename T>
	tensor<T>* operator+(const T& left, const tensor<T>& right)
	{
		tensor<T>* child = createTensorWithSameDims(right);

		// make kernel call
		tensorAdd(left, right, *child);

		return child;
	}

	template<typename T>
	tensor<T>* operator-(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		tensor<T>* child = createTensorWithSameDims(left, right);

		// make kernel call
		tensorSubtract(left, right, *child);

		return child;
	}

	template<typename T>
	tensor<T>* operator-(const tensor<T>& left, const T& right)
	{
		T q = -right;
		return left + q;
	}

	template<typename T>
	tensor<T>* operator-(const T& left, const tensor<T>& right)
	{
		tensor<T>* child = createTensorWithSameDims(right);

		// make kernel call
		tensorSubtract(left, right, *child);

		return child;
	}

	template<typename T>
	tensor<T>* operator-(const tensor<T>& left)
	{
		const T right = { 0 };
		return right - left;
	}

	template<typename T>
	tensor<T>* permute(const tensor<T>& inTensor, std::vector<uint32_t> dims)
	{

		if (dims.size() != inTensor.getDimensions().size())
		{
			throw std::exception("cannot perform permutation due to incorrect number of permute dimensions.");
		}
		for (uint32_t i = 0; i < inTensor.getDimensions().size(); i++)
		{
			if (dims[i] >= inTensor.getDimensions().size())
			{
				throw std::exception("permute dimensions indicies cannot be higher than tensor dimnesion count.");
			}
		}

		tensor<T>* child = createTensorWithPermutedDims(inTensor, dims);

		// make kernel call
		tensorPermute(inTensor, *child, dims);

		return child;
	}
}
