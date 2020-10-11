#include <cstdint>
#include "tensor.h"

namespace dnnbasic
{
	template<typename From, typename To>
	void tensorCast(const tensor<From>& from, tensor<To>& to);
}