#include "kernel_tools.h"

namespace dnnbasic
{
	int integerCeilDivision(int a, int b)
	{
		//return (int) math.ceil((float)a / b);
		return (a + (b - 1)) / b;
	}
}