#include <CppUnitTest.h>
#include <string>
#include "tensor.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorMultiDimMatMulTests)
	{
	public:
		template<typename T>
		void matrixMatrix3x2MulMatrix2x3()
		{
			dnnbasic::tensor<T> a({ 2, 3, 2 },
				{
					5,7,
					4,8,
					6,1,

					5,7,
					4,8,
					6,1
				});
			dnnbasic::tensor<T> b({ 2, 2, 3 },
				{
					7,5,4,
					7,9,6,

					7,5,4,
					7,9,6
				});

			dnnbasic::tensor<T> expected({ 2, 3, 3 },
				{
					84,88,62,
					84,92,64,
					49,39,30,

					84,88,62,
					84,92,64,
					49,39,30
				});
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixMatrix3x2MulMatrix2x3)
	};
}