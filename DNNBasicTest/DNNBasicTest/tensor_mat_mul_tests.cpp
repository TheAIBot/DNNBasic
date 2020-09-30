#include <CppUnitTest.h>
#include <string>
#include "tensor.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorMatrixOpTests)
	{
	public:
		template<typename T>
		void matrix2x2MulMatrix2x2()
		{
			dnnbasic::tensor<T> a({ 2, 2 }, 
				{ 
					1,0,
					0,1 
				});
			dnnbasic::tensor<T> b({ 2, 2 }, 
				{ 
					4,1,
					1,2 
				});

			dnnbasic::tensor<T> expected({ 2, 2 }, 
				{ 
					4,1,
					1,2 
				});
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrix2x2MulMatrix2x2)

		template<typename T>
		void matrix3x2MulMatrix2x3()
		{
			dnnbasic::tensor<T> a({ 3, 2 },
				{
					5,7,
					4,8,
					6,1
				});
			dnnbasic::tensor<T> b({ 2, 3 },
				{
					7,5,4,
					7,9,6
				});

			dnnbasic::tensor<T> expected({ 3, 3 },
				{
					84,88,62,
					84,92,64,
					49,39,30
				});
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrix3x2MulMatrix2x3)

		template<typename T>
		void matrixMulVector()
		{
			dnnbasic::tensor<T> a({ 3, 2 },
				{
					5,7,
					4,8,
					6,1
				});
			dnnbasic::tensor<T> b({ 2 },
				{
					7,
					7
				});

			dnnbasic::tensor<T> expected({ 3 },
				{
					84,
					84,
					49
				});
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixMulVector)

		template<typename T>
		void vectorMulMatrix()
		{
			dnnbasic::tensor<T> a({ 2 },
				{
					5, 7
				});
			dnnbasic::tensor<T> b({ 2, 3 },
				{
					7, 5, 4,
					7, 9, 6
				});

			dnnbasic::tensor<T> expected({ 3 },
				{
					84, 88, 62
				});
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(vectorMulMatrix)
	};
}