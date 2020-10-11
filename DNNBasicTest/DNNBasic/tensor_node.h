#pragma once

#include "FBPropagation.h"

namespace dnnbasic
{
	template<typename T>
	class tensorNode : public fbpropagation<T>
	{
	private:
		tensor<T> forward(const tensor<T>& x) const override
		{
			throw new std::runtime_error("Wait how you do that?");
		}
	};
}