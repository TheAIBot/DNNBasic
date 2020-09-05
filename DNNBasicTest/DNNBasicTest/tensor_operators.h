#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "span.h"
#include "tensor_def.h"

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
	static tensor<T>* createTensorWithSameDims(tensor<T>& a, tensor<T>& b)
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
	bool operator==(tensor<T>& left, tensor<T>& right)
	{
		const auto& leftC = left;
		const auto& rightC = right;
		return leftC == rightC;
	}

	template<typename T>
	bool operator!=(tensor<T>& left, tensor<T>& right)
	{
		return !(left == right);
	}

	template<typename T>
	bool operator==(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		for (size_t i = 0; i < left.arr.size(); i++)
		{
			if (left.arr[i] != right.arr[i])
			{
				return false;
			}
		}

		return true;
	}

	template<typename T>
	bool operator!=(const tensor<T>& left, const tensor<T>& right)
	{
		return !(left == right);
	}


	template<typename T>
	tensor<T>* operator*(tensor<T>& left, tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		tensor<T>* child = createTensorWithSameDims(left, right);
		child->addConnection(&left);
		child->addConnection(&right);

		// make kernel call
		for (uint32_t i = 0; i < child->elementCount(); i++)
		{
			(*child)[i] = left[i] * right[i];
		}

		return child;
	}

}