#pragma once
#include "cudaStream.h"

namespace dnnbasic
{
	namespace cuda
	{
		cudaStream_t getDefaultStream();
	}
}