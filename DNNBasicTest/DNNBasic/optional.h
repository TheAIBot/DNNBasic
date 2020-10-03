#pragma once
#include <assert.h>
namespace dnnbasic 
{
	template<typename T>
	class optional
	{
	private:
		T vvalue;
		bool hasValue;
	public:
		optional(T& val) : vvalue(val)
		{
			this->hasValue = true;
		}
		optional()
		{
			this->hasValue = false;
		}

		bool has_value() const
		{
			return this->hasValue;
		}

		T value() const
		{
			assert(hasValue);
			return vvalue;
		}
	};

}