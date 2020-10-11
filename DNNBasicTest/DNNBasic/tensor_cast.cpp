#include <vector>
#include <string>
#include "tensor_cast_kernel.cuh"
#include "tensor.h"
#include "tensor_node_no_grad.h"

namespace dnnbasic
{
	template<typename From>
	template<typename To>
	tensor<To> tensor<From>::cast() const
	{
		if constexpr (std::is_same<From, To>::value)
		{
			return *this;
		}
		else
		{
			auto dims = this->getDimensions();

			std::vector<uint32_t> newDims;
			std::vector<std::string> newDimNames;

			for (size_t i = 0; i < dims.size(); i++)
			{
				newDims.push_back(dims[i].dim);
				newDimNames.push_back(dims[i].name);
			}

			tensor<To> to(newDims, newDimNames);

			tensorCast(*this, to);

			return to;
		}
	}

#define CAST_FROM_TO(fromTyp, toTyp) \
	template tensor<toTyp> tensor<fromTyp>::cast() const;

#define CAST_FROM(fromTyp) \
	CAST_FROM_TO(fromTyp, bool) \
	CAST_FROM_TO(fromTyp, uint8_t) \
	CAST_FROM_TO(fromTyp, uint16_t) \
	CAST_FROM_TO(fromTyp, uint32_t) \
	CAST_FROM_TO(fromTyp, uint64_t) \
	CAST_FROM_TO(fromTyp, int8_t) \
	CAST_FROM_TO(fromTyp, int16_t) \
	CAST_FROM_TO(fromTyp, int32_t) \
	CAST_FROM_TO(fromTyp, int64_t) \
	CAST_FROM_TO(fromTyp, float) \
	CAST_FROM_TO(fromTyp, double)

	CAST_FROM(bool)
	CAST_FROM(uint8_t)
	CAST_FROM(uint16_t)
	CAST_FROM(uint32_t)
	CAST_FROM(uint64_t)
	CAST_FROM(int8_t)
	CAST_FROM(int16_t)
	CAST_FROM(int32_t)
	CAST_FROM(int64_t)
	CAST_FROM(float)
	CAST_FROM(double)
}