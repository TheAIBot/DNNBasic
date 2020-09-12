#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "tensor_def.h"
#include "tensor_matrix_kernels.cuh"

namespace dnnbasic
{
	template<typename T>
	bool canMatrixMultiplyMatrix(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		if (aDims.size() == 2 && bDims.size() == 2 &&
			aDims[1].dim == bDims[0].dim)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename T>
	bool canMatrixMultiplyVector(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		if (aDims.size() == 2 && bDims.size() == 1 &&
			aDims[1].dim == bDims[0].dim)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename T>
	bool canVectorMultiplyMatrix(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		if (aDims.size() == 1 && bDims.size() == 2 &&
			aDims[0].dim == bDims[0].dim)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename T>
	tensor<T>* createTensorWithMatrixMultiplyMatrixDims(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;

		new_dim.push_back(aDims[0].dim);
		new_dim.push_back(bDims[1].dim);

		new_name.push_back(aDims.front().name != "" ? aDims[0].name : bDims[0].name);
		new_name.push_back(aDims.front().name != "" ? aDims[1].name : bDims[1].name);

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T>* createTensorWithMatrixMultiplyVectorDims(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;

		new_dim.push_back(aDims[0].dim);
		new_name.push_back(bDims[0].name);

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T>* createTensorWithVectorMultiplyMatrixDims(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;

		new_dim.push_back(bDims[1].dim);
		new_name.push_back(aDims[0].name);

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T>* matMul(const tensor<T>& left, const tensor<T>& right)
	{
		if (canMatrixMultiplyMatrix(left, right))
		{
			tensor<T>* child = createTensorWithMatrixMultiplyMatrixDims(left, right);

			matrix<T> leftM = left.getMatrix();
			matrix<T> rightM = right.getMatrix();
			matrix<T> childM = child->getMatrix();

			// make kernel call
			tensorMatrixMul(leftM, rightM, childM);

			return child;
		}
		else if (canMatrixMultiplyVector(left, right))
		{
			tensor<T>* child = createTensorWithMatrixMultiplyVectorDims(left, right);

			matrix<T> leftM = left.getMatrix();
			matrix<T> rightM = right.getMatrixWith1Width();
			matrix<T> childM = child->getMatrixWith1Width();

			// make kernel call
			tensorMatrixMul(leftM, rightM, childM);

			return child;
		}
		else if (canVectorMultiplyMatrix(left, right))
		{
			tensor<T>* child = createTensorWithVectorMultiplyMatrixDims(left, right);

			matrix<T> leftM = left.getMatrixWith1Height();
			matrix<T> rightM = right.getMatrix();
			matrix<T> childM = child->getMatrixWith1Height();

			// make kernel call
			tensorMatrixMul(leftM, rightM, childM);

			return child;
		}
		else
		{
			throw std::exception("Left hand side tensor cannot matrix multiply with right hand side tensor.");
		}

	}
}