#pragma once

#include <vector>
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
	public:
		virtual std::vector<tensor<T>> getTensors() const = 0;
	};
}