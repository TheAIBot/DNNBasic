#pragma once

#include "tensor.h"
#include <cstdint>
#include <vector>
#include <string>

namespace dnnbasic
{
	template<typename T>
	bool canBroadcastTensors(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		int32_t aDimsIdx = (int32_t)a.getDimensions().size() - 1;
		int32_t bDimsIdx = (int32_t)b.getDimensions().size() - 1;

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
	tensor<T> createBroadcastedTensor(const tensor<T>& a, const tensor<T>& b)
	{
		auto& aDims = a.getDimensions();
		auto& bDims = b.getDimensions();

		std::vector<uint32_t> new_dim(std::max(aDims.size(), bDims.size()));
		std::vector<std::string> new_name(new_dim.size());

		int32_t aDimsIdx = (int32_t)a.getDimensions().size() - 1;
		int32_t bDimsIdx = (int32_t)b.getDimensions().size() - 1;

		for (int32_t i = (int32_t)new_dim.size() - 1; i >= 0; i--)
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

		return tensor<T>(new_dim, new_name);
	}
}