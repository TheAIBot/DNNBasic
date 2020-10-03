#include <string>
#include <vector>
#include <cstdint>
#include "span.h"
#include "tensor.h"
#include "tensor_elementwise_kernels.cuh"
#include "auto_graph.h"
#include "tensor_node_no_grad.h"

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
	static tensor<T> createTensorWithSameDims(const tensor<T>& a, const tensor<T>& b)
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

		return tensor<T>(new_dim, new_name);
	}

	template<typename T>
	static tensor<T> createTensorWithSameDims(const tensor<T>& a)
	{
		auto& aDims = a.getDimensions();

		std::vector<uint32_t> new_dim;
		std::vector<std::string> new_name;
		for (size_t i = 0; i < aDims.size(); i++)
		{
			new_dim.push_back(aDims[i].dim);
			new_name.push_back(aDims[i].name);
		}

		return tensor<T>(new_dim, new_name);
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
	tensor<T> operator*(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		tensor<T> child = createTensorWithSameDims(left, right);
		if (autoGraph::makeGraph)
		{
			child.setNode(new tensorNodeNoGrad<T>({ left, right }));
		}

		// make kernel call
		tensorMultiply(left, right, child);

		return child;
	}

	template<typename T>
	tensor<T> operator*(const tensor<T>& left, const T& right)
	{
		return right * left;
	}

	template<typename T>
	tensor<T> operator*(const T& left, const tensor<T>& right)
	{
		tensor<T> child = createTensorWithSameDims(right);
		if (autoGraph::makeGraph)
		{
			child.setNode(new tensorNodeNoGrad<T>({ right }));
		}

		// make kernel call
		tensorMultiply(left, right, child);

		return child;
	}

	template<typename T>
	tensor<T> operator+(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		tensor<T> child = createTensorWithSameDims(left, right);
		if (autoGraph::makeGraph)
		{
			child.setNode(new tensorNodeNoGrad<T>({ left, right }));
		}

		// make kernel call
		tensorAdd(left, right, child);

		return child;
	}

	template<typename T>
	tensor<T> operator+(const tensor<T>& left, const T& right)
	{
		return right + left;
	}

	template<typename T>
	tensor<T> operator+(const T& left, const tensor<T>& right)
	{
		tensor<T> child = createTensorWithSameDims(right);
		if (autoGraph::makeGraph)
		{
			child.setNode(new tensorNodeNoGrad<T>({right}));
		}

		// make kernel call
		tensorAdd(left, right, child);

		return child;
	}

	template<typename T>
	tensor<T> operator-(const tensor<T>& left, const tensor<T>& right)
	{
		if (!hasSameDimensions(left, right))
		{
			throw std::exception("Dimensions of left hand side tensor do not match dimension of right hand side tensor.");
		}

		tensor<T> child = createTensorWithSameDims(left, right);
		if (autoGraph::makeGraph)
		{
			child.setNode(new tensorNodeNoGrad<T>({ left, right }));
		}
		// make kernel call
		tensorSubtract(left, right, child);

		return child;
	}

	template<typename T>
	tensor<T> operator-(const tensor<T>& left, const T& right)
	{
		T q = -right;
		return left + q;
	}

	template<typename T>
	tensor<T> operator-(const T& left, const tensor<T>& right)
	{
		tensor<T> child = createTensorWithSameDims(right);
		if (autoGraph::makeGraph)
		{
			child.setNode(new tensorNodeNoGrad<T>({ right }));
		}

		// make kernel call
		tensorSubtract(left, right, child);

		return child;
	}

	template<typename T>
	tensor<T> operator-(const tensor<T>& left)
	{
		const T right = { 0 };
		return right - left;
	}


	template bool operator==(const tensor<bool>& left, const tensor<bool>& right);
	template bool operator==(const tensor<uint8_t>& left, const tensor<uint8_t>& right);
	template bool operator==(const tensor<uint16_t>& left, const tensor<uint16_t>& right);
	template bool operator==(const tensor<uint32_t>& left, const tensor<uint32_t>& right);
	template bool operator==(const tensor<uint64_t>& left, const tensor<uint64_t>& right);
	template bool operator==(const tensor<int8_t>& left, const tensor<int8_t>& right);
	template bool operator==(const tensor<int16_t>& left, const tensor<int16_t>& right);
	template bool operator==(const tensor<int32_t>& left, const tensor<int32_t>& right);
	template bool operator==(const tensor<int64_t>& left, const tensor<int64_t>& right);
	template bool operator==(const tensor<float>& left, const tensor<float>& right);
	template bool operator==(const tensor<double>& left, const tensor<double>& right);

	template bool operator!=(const tensor<bool>& left, const tensor<bool>& right);
	template bool operator!=(const tensor<uint8_t>& left, const tensor<uint8_t>& right);
	template bool operator!=(const tensor<uint16_t>& left, const tensor<uint16_t>& right);
	template bool operator!=(const tensor<uint32_t>& left, const tensor<uint32_t>& right);
	template bool operator!=(const tensor<uint64_t>& left, const tensor<uint64_t>& right);
	template bool operator!=(const tensor<int8_t>& left, const tensor<int8_t>& right);
	template bool operator!=(const tensor<int16_t>& left, const tensor<int16_t>& right);
	template bool operator!=(const tensor<int32_t>& left, const tensor<int32_t>& right);
	template bool operator!=(const tensor<int64_t>& left, const tensor<int64_t>& right);
	template bool operator!=(const tensor<float>& left, const tensor<float>& right);
	template bool operator!=(const tensor<double>& left, const tensor<double>& right);

	//template tensor<bool> operator*(const tensor<bool>& left, const tensor<bool>& right);
	template tensor<uint8_t> operator*(const tensor<uint8_t>& left, const tensor<uint8_t>& right);
	template tensor<uint16_t> operator*(const tensor<uint16_t>& left, const tensor<uint16_t>& right);
	template tensor<uint32_t> operator*(const tensor<uint32_t>& left, const tensor<uint32_t>& right);
	template tensor<uint64_t> operator*(const tensor<uint64_t>& left, const tensor<uint64_t>& right);
	template tensor<int8_t> operator*(const tensor<int8_t>& left, const tensor<int8_t>& right);
	template tensor<int16_t> operator*(const tensor<int16_t>& left, const tensor<int16_t>& right);
	template tensor<int32_t> operator*(const tensor<int32_t>& left, const tensor<int32_t>& right);
	template tensor<int64_t> operator*(const tensor<int64_t>& left, const tensor<int64_t>& right);
	template tensor<float> operator*(const tensor<float>& left, const tensor<float>& right);
	template tensor<double> operator*(const tensor<double>& left, const tensor<double>& right);

	//template tensor<bool> operator*(const tensor<bool>& left, const bool& right);
	template tensor<uint8_t> operator*(const tensor<uint8_t>& left, const uint8_t& right);
	template tensor<uint16_t> operator*(const tensor<uint16_t>& left, const uint16_t& right);
	template tensor<uint32_t> operator*(const tensor<uint32_t>& left, const uint32_t& right);
	template tensor<uint64_t> operator*(const tensor<uint64_t>& left, const uint64_t& right);
	template tensor<int8_t> operator*(const tensor<int8_t>& left, const int8_t& right);
	template tensor<int16_t> operator*(const tensor<int16_t>& left, const int16_t& right);
	template tensor<int32_t> operator*(const tensor<int32_t>& left, const int32_t& right);
	template tensor<int64_t> operator*(const tensor<int64_t>& left, const int64_t& right);
	template tensor<float> operator*(const tensor<float>& left, const float& right);
	template tensor<double> operator*(const tensor<double>& left, const double& right);

	//template tensor<bool> operator*(const bool& left, const tensor<bool>& right);
	template tensor<uint8_t> operator*(const uint8_t& left, const tensor<uint8_t>& right);
	template tensor<uint16_t> operator*(const uint16_t& left, const tensor<uint16_t>& right);
	template tensor<uint32_t> operator*(const uint32_t& left, const tensor<uint32_t>& right);
	template tensor<uint64_t> operator*(const uint64_t& left, const tensor<uint64_t>& right);
	template tensor<int8_t> operator*(const int8_t& left, const tensor<int8_t>& right);
	template tensor<int16_t> operator*(const int16_t& left, const tensor<int16_t>& right);
	template tensor<int32_t> operator*(const int32_t& left, const tensor<int32_t>& right);
	template tensor<int64_t> operator*(const int64_t& left, const tensor<int64_t>& right);
	template tensor<float> operator*(const float& left, const tensor<float>& right);
	template tensor<double> operator*(const double& left, const tensor<double>& right);


	//template tensor<bool> operator+(const tensor<bool>& left, const tensor<bool>& right);
	template tensor<uint8_t> operator+(const tensor<uint8_t>& left, const tensor<uint8_t>& right);
	template tensor<uint16_t> operator+(const tensor<uint16_t>& left, const tensor<uint16_t>& right);
	template tensor<uint32_t> operator+(const tensor<uint32_t>& left, const tensor<uint32_t>& right);
	template tensor<uint64_t> operator+(const tensor<uint64_t>& left, const tensor<uint64_t>& right);
	template tensor<int8_t> operator+(const tensor<int8_t>& left, const tensor<int8_t>& right);
	template tensor<int16_t> operator+(const tensor<int16_t>& left, const tensor<int16_t>& right);
	template tensor<int32_t> operator+(const tensor<int32_t>& left, const tensor<int32_t>& right);
	template tensor<int64_t> operator+(const tensor<int64_t>& left, const tensor<int64_t>& right);
	template tensor<float> operator+(const tensor<float>& left, const tensor<float>& right);
	template tensor<double> operator+(const tensor<double>& left, const tensor<double>& right);

	//template tensor<bool> operator+(const tensor<bool>& left, const bool& right);
	template tensor<uint8_t> operator+(const tensor<uint8_t>& left, const uint8_t& right);
	template tensor<uint16_t> operator+(const tensor<uint16_t>& left, const uint16_t& right);
	template tensor<uint32_t> operator+(const tensor<uint32_t>& left, const uint32_t& right);
	template tensor<uint64_t> operator+(const tensor<uint64_t>& left, const uint64_t& right);
	template tensor<int8_t> operator+(const tensor<int8_t>& left, const int8_t& right);
	template tensor<int16_t> operator+(const tensor<int16_t>& left, const int16_t& right);
	template tensor<int32_t> operator+(const tensor<int32_t>& left, const int32_t& right);
	template tensor<int64_t> operator+(const tensor<int64_t>& left, const int64_t& right);
	template tensor<float> operator+(const tensor<float>& left, const float& right);
	template tensor<double> operator+(const tensor<double>& left, const double& right);

	//template tensor<bool> operator+(const bool& left, const tensor<bool>& right);
	template tensor<uint8_t> operator+(const uint8_t& left, const tensor<uint8_t>& right);
	template tensor<uint16_t> operator+(const uint16_t& left, const tensor<uint16_t>& right);
	template tensor<uint32_t> operator+(const uint32_t& left, const tensor<uint32_t>& right);
	template tensor<uint64_t> operator+(const uint64_t& left, const tensor<uint64_t>& right);
	template tensor<int8_t> operator+(const int8_t& left, const tensor<int8_t>& right);
	template tensor<int16_t> operator+(const int16_t& left, const tensor<int16_t>& right);
	template tensor<int32_t> operator+(const int32_t& left, const tensor<int32_t>& right);
	template tensor<int64_t> operator+(const int64_t& left, const tensor<int64_t>& right);
	template tensor<float> operator+(const float& left, const tensor<float>& right);
	template tensor<double> operator+(const double& left, const tensor<double>& right);


	//template tensor<bool> operator-(const tensor<bool>& left, const tensor<bool>& right);
	template tensor<uint8_t> operator-(const tensor<uint8_t>& left, const tensor<uint8_t>& right);
	template tensor<uint16_t> operator-(const tensor<uint16_t>& left, const tensor<uint16_t>& right);
	template tensor<uint32_t> operator-(const tensor<uint32_t>& left, const tensor<uint32_t>& right);
	template tensor<uint64_t> operator-(const tensor<uint64_t>& left, const tensor<uint64_t>& right);
	template tensor<int8_t> operator-(const tensor<int8_t>& left, const tensor<int8_t>& right);
	template tensor<int16_t> operator-(const tensor<int16_t>& left, const tensor<int16_t>& right);
	template tensor<int32_t> operator-(const tensor<int32_t>& left, const tensor<int32_t>& right);
	template tensor<int64_t> operator-(const tensor<int64_t>& left, const tensor<int64_t>& right);
	template tensor<float> operator-(const tensor<float>& left, const tensor<float>& right);
	template tensor<double> operator-(const tensor<double>& left, const tensor<double>& right);

	//template tensor<bool> operator-(const tensor<bool>& left, const bool& right);
	//template tensor<uint8_t> operator-(const tensor<uint8_t>& left, const uint8_t& right);
	//template tensor<uint16_t> operator-(const tensor<uint16_t>& left, const uint16_t& right);
	//template tensor<uint32_t> operator-(const tensor<uint32_t>& left, const uint32_t& right);
	//template tensor<uint64_t> operator-(const tensor<uint64_t>& left, const uint64_t& right);
	template tensor<int8_t> operator-(const tensor<int8_t>& left, const int8_t& right);
	template tensor<int16_t> operator-(const tensor<int16_t>& left, const int16_t& right);
	template tensor<int32_t> operator-(const tensor<int32_t>& left, const int32_t& right);
	template tensor<int64_t> operator-(const tensor<int64_t>& left, const int64_t& right);
	template tensor<float> operator-(const tensor<float>& left, const float& right);
	template tensor<double> operator-(const tensor<double>& left, const double& right);

	//template tensor<bool> operator-(const bool& left, const tensor<bool>& right);
	template tensor<uint8_t> operator-(const uint8_t& left, const tensor<uint8_t>& right);
	template tensor<uint16_t> operator-(const uint16_t& left, const tensor<uint16_t>& right);
	template tensor<uint32_t> operator-(const uint32_t& left, const tensor<uint32_t>& right);
	template tensor<uint64_t> operator-(const uint64_t& left, const tensor<uint64_t>& right);
	template tensor<int8_t> operator-(const int8_t& left, const tensor<int8_t>& right);
	template tensor<int16_t> operator-(const int16_t& left, const tensor<int16_t>& right);
	template tensor<int32_t> operator-(const int32_t& left, const tensor<int32_t>& right);
	template tensor<int64_t> operator-(const int64_t& left, const tensor<int64_t>& right);
	template tensor<float> operator-(const float& left, const tensor<float>& right);
	template tensor<double> operator-(const double& left, const tensor<double>& right);

	//template tensor<bool> operator-(const tensor<bool>& left);
	template tensor<uint8_t> operator-(const tensor<uint8_t>& left);
	template tensor<uint16_t> operator-(const tensor<uint16_t>& left);
	template tensor<uint32_t> operator-(const tensor<uint32_t>& left);
	template tensor<uint64_t> operator-(const tensor<uint64_t>& left);
	template tensor<int8_t> operator-(const tensor<int8_t>& left);
	template tensor<int16_t> operator-(const tensor<int16_t>& left);
	template tensor<int32_t> operator-(const tensor<int32_t>& left);
	template tensor<int64_t> operator-(const tensor<int64_t>& left);
	template tensor<float> operator-(const tensor<float>& left);
	template tensor<double> operator-(const tensor<double>& left);
}
