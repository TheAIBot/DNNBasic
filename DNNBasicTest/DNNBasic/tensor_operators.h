#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "span.h"
#include "tensor_def.h"
#include "tensor_elementwise_kernels.cuh"
#include "tensor_matrix_kernels.cuh"

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
