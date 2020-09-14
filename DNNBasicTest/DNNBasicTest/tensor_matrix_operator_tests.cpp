#include <CppUnitTest.h>
#include <string>
#include <typeinfo>
#include <codecvt>
#include "Tensor.h"
#include "test_tools.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DNNBasicTest
{
	TEST_CLASS(tensorMatrixOpTests)
	{
	public:
		template<typename T>
		void matrixMatrix2x2MulMatrix2x2()
		{
			dnnbasic::tensor<T> a({ 2, 2 }, { 1,0,0,1 });
			dnnbasic::tensor<T> b({ 2, 2 }, { 4,1,1,2 });

			dnnbasic::tensor<T> expected({ 2, 2 }, { 4,1,1,2 });
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixMatrix2x2MulMatrix2x2)

			template<typename T>
		void matrixMatrix3x2MulMatrix2x3()
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
		TEST_ALL_OP_TYPES(matrixMatrix3x2MulMatrix2x3)

			template<typename T>
		void matrixMatrixMulVector()
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
		TEST_ALL_OP_TYPES(matrixMatrixMulVector)

			template<typename T>
		void matrixVectorMulMatrix()
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
		TEST_ALL_OP_TYPES(matrixVectorMulMatrix)
	};
}