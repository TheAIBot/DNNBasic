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
		void matrixMutiDimMatrix3x2MulMatrix2x3()
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
		TEST_ALL_OP_TYPES(matrixMutiDimMatrix3x2MulMatrix2x3)

			template<typename T>
		void matrixMultiDimMatrixMul4Dim()
		{
			dnnbasic::tensor<T> a({ 2, 2, 3, 2 },
				{
					0,1,
					2,3,
					4,5,

					6,7,
					8,9,
					10,11,


					12,13,
					14,15,
					16,17,

					18,19,
					20,21,
					22,23
				});
			dnnbasic::tensor<T> b({ 2, 2, 2, 3 },
				{
					0,1,2,
					3,4,5,

					6,7,8,
					9,10,11,


					12,13,14,
					15,16,17,

					18,19,20,
					21,22,23
				});

			dnnbasic::tensor<T> expected({2, 2, 3, 3 },
				{
					3,4,5,
					9,14,19,
					15,24,33,

					99,112,125,
					129,146,163,
					159,180,201,

					339,364,389,
					393,422,451,
					447,480,513,

					723,760,797,
					801,842,883,
					879,924,969

				});
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES_LEAST_8_BITS(matrixMultiDimMatrixMul4Dim)

			template<typename T>
		void matrixMutiDimVectorMulMatrix()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 },
				{
					69, 108, 134,

					72, 109, 147
				});
			dnnbasic::tensor<T> b({ 2, 3, 3 },
				{

					18,10,6,
					8,9,16,
					13,5,10,

					6,6,15,
					3,8,5,
					3,15,7
				});

			dnnbasic::tensor<T> expected({ 2, 1, 3 },
				{
					3848, 2332, 3482,

					1200, 3509, 2654
				});
			auto* actual = a.matMul(b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES_LEAST_8_BITS(matrixMutiDimVectorMulMatrix)
	};
}