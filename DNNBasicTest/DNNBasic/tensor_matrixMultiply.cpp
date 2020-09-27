#include <string>
#include <vector>
#include <cstdint>
#include "tensor.h"
#include "tensor_matrix_kernels.cuh"
#include "tensor_multi_dim_matrix_mul.cuh"

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
	bool canBroadcastMatrixMultiplyBroadcastMatrix(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		if (!(aDims.size() > 1 && bDims.size() > 1))
		{
			return false;
		}

		if (aDims[aDims.size() - 1].dim != bDims[bDims.size() - 2].dim)
		{
			return false;
		}

		const uint32_t matrixDimsCount = 2;
		int32_t aDimsIdx = (int32_t)a.getDimensions().size() - 1 - matrixDimsCount;
		int32_t bDimsIdx = (int32_t)b.getDimensions().size() - 1 - matrixDimsCount;

		while (true)
		{
			if (aDimsIdx < 0 || bDimsIdx < 0)
			{
				return true;
			}
			else if (aDims[aDimsIdx].dim == bDims[bDimsIdx].dim)
			{
				
			}
			else if (aDims[aDimsIdx].dim == 1 || bDims[bDimsIdx].dim == 1)
			{
				
			}
			else
			{
				return false;
			}

			aDimsIdx--;
			bDimsIdx--;
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

		new_name.push_back(aDims[aDims.size() - 2].name != "" ? aDims[0].name : bDims[0].name);
		new_name.push_back(aDims[aDims.size() - 1].name != "" ? aDims[1].name : bDims[1].name);

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
	tensor<T>* createTensorWithBroadcastMatrixMultiplyBroadcastMatrixDims(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		std::vector<uint32_t> new_dim(std::max(aDims.size(), bDims.size()));
		std::vector<std::string> new_name(new_dim.size());

		const uint32_t matrixDimsCount = 2;
		int32_t aDimsIdx = (int32_t)a.getDimensions().size() - 1 - matrixDimsCount;
		int32_t bDimsIdx = (int32_t)b.getDimensions().size() - 1 - matrixDimsCount;

		for (int32_t i = (int32_t)new_dim.size() - 1 - matrixDimsCount; i >= 0; i--)
		{
			if (bDimsIdx < 0)
			{
				new_dim[i] = aDims[aDimsIdx].dim;
				new_name[i] = aDims[aDimsIdx].name;
			}
			else if (aDimsIdx < 0)
			{
				new_dim[i] = bDims[bDimsIdx].dim;
				new_name[i] = bDims[bDimsIdx].name;
			}
			else
			{
				new_dim[i] = std::max(aDims[aDimsIdx].dim, bDims[bDimsIdx].dim);
				new_name[i] = aDims[aDimsIdx].name != "" ? aDims[aDimsIdx].name : bDims[bDimsIdx].name;
			}
			aDimsIdx--;
			bDimsIdx--;
		}

		new_dim[new_dim.size() - 2] = aDims[aDims.size() - 2].dim;
		new_dim[new_dim.size() - 1] = bDims[bDims.size() - 1].dim;

		new_name[new_dim.size() - 2] = aDims[aDims.size() - 2].name != "" ? aDims[aDims.size() - 2].name : bDims[bDims.size() - 2].name;
		new_name[new_dim.size() - 1] = aDims[aDims.size() - 1].name != "" ? aDims[aDims.size() - 1].name : bDims[bDims.size() - 1].name;

		return new tensor<T>(new_dim, new_name);
	}

	template<typename T>
	tensor<T>* tensor<T>::matMul(const tensor<T>& right) const
	{
		if (canMatrixMultiplyMatrix(*this, right))
		{
			tensor<T>* child = createTensorWithMatrixMultiplyMatrixDims(*this, right);

			matrix<T> leftM = this->getMatrix();
			matrix<T> rightM = right.getMatrix();
			matrix<T> childM = child->getMatrix();

			// make kernel call
			tensorMatrixMul(leftM, rightM, childM);

			return child;
		}
		else if (canMatrixMultiplyVector(*this, right))
		{
			tensor<T>* child = createTensorWithMatrixMultiplyVectorDims(*this, right);

			matrix<T> leftM = this->getMatrix();
			matrix<T> rightM = right.getMatrixWith1Width();
			matrix<T> childM = child->getMatrixWith1Width();

			// make kernel call
			tensorMatrixMul(leftM, rightM, childM);

			return child;
		}
		else if (canVectorMultiplyMatrix(*this, right))
		{
			tensor<T>* child = createTensorWithVectorMultiplyMatrixDims(*this, right);

			matrix<T> leftM = this->getMatrixWith1Height();
			matrix<T> rightM = right.getMatrix();
			matrix<T> childM = child->getMatrixWith1Height();

			// make kernel call
			tensorMatrixMul(leftM, rightM, childM);

			return child;
		}
		else if (canBroadcastMatrixMultiplyBroadcastMatrix(*this, right))
		{
			tensor<T>* child = createTensorWithBroadcastMatrixMultiplyBroadcastMatrixDims(*this, right);

			// make kernel call
			tensorMultiDimMatrixMul(*this, right, *child);

			return child;
		}
		else
		{
			throw std::exception("Left hand side tensor cannot matrix multiply with right hand side tensor.");
		}

	}
}