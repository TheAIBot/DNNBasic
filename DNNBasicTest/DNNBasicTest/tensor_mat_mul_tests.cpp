#include <CppUnitTest.h>
#include <string>
#include <vector>
#include <array>
#include "tensor.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorMatMulTests)
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
			auto actual = a.matMul(b);

			Assert::AreEqual(expected, actual);
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
			auto actual = a.matMul(b);

			Assert::AreEqual(expected, actual);
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
			auto actual = a.matMul(b);

			Assert::AreEqual(expected, actual);
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
			auto actual = a.matMul(b);

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(vectorMulMatrix)
	};

	TEST_CLASS(tensorMatMulTestsAutoGenerated)
	{
		static constexpr std::array<uint32_t, 6> values = { 1,2, 3, 33, 174, 643 };

		template<typename T>
		static dnnbasic::matrix<T> transpose(dnnbasic::matrix<T> a, std::vector<T>& tData)
		{
			dnnbasic::matrix<T> t(&tData[0], a.getRows(), a.getColumns());
			for (uint32_t y = 0; y < a.getRows(); y++)
			{
				for (uint32_t x = 0; x < a.getColumns(); x++)
				{
					t[x][y] = a[y][x];
				}
			}

			return t;
		}

		template<typename T>
		static void matMulCPU(dnnbasic::matrix<T> a, dnnbasic::matrix<T> b, dnnbasic::matrix<T> c)
		{
			std::vector<T> tData(b.getColumns() * b.getRows());
			b = transpose(b, tData);
			for (uint32_t y = 0; y < c.getRows(); y++)
			{
				for (uint32_t x = 0; x < c.getColumns(); x++)
				{
					T sum = 0;
					for (uint32_t z = 0; z < a.getColumns(); z++)
					{
						sum += a[y][z] * b[x][z];
					}
					c[y][x] = sum;
				}
			}
		}

		template<typename T>
		static void matMulTest(uint32_t aWidth, uint32_t aHeight, uint32_t bWidth, uint32_t bHeight)
		{
			dnnbasic::tensor<T> a = dnnbasic::tensor<T>::random({ aHeight, aWidth });
			dnnbasic::tensor<T> b = dnnbasic::tensor<T>::random({ bHeight, bWidth });

			auto aData = a.getValuesOnCPU();
			auto bData = b.getValuesOnCPU();
			std::vector<T> cData(aHeight * bWidth);

			dnnbasic::matrix<T> aMatrix(&aData[0], aWidth, aHeight);
			dnnbasic::matrix<T> bMatrix(&bData[0], bWidth, bHeight);
			dnnbasic::matrix<T> cMatrix(&cData[0], bWidth, aHeight);

			matMulCPU(aMatrix, bMatrix, cMatrix);


			dnnbasic::tensor<T> expected({ cMatrix.getRows(), cMatrix.getColumns() }, cData);

			assertCloseEnoughTensorOp<T>(expected, [&]() {return a.matMul(b); });
		}
	public:

#define genTest(a, b, c, T) \
	TEST_METHOD(T ## mathMul_ ## a ## _ ## b ## _ ## c) { matMulTest<T>( ## values[b] ## , ## values[a] ## , ## values[c] ## , ## values[b] ## ); }

#define genTest1(a, b, T) \
	genTest(a, b, 0, T) \
	genTest(a, b, 1, T) \
	genTest(a, b, 2, T) \
	genTest(a, b, 3, T) \
	genTest(a, b, 4, T) \
	genTest(a, b, 5, T)

#define genTest2(a, T) \
	genTest1(a, 0, T) \
	genTest1(a, 1, T) \
	genTest1(a, 2, T) \
	genTest1(a, 3, T) \
	genTest1(a, 4, T) \
	genTest1(a, 5, T)

#define genTest3(T) \
	genTest2(0, T) \
	genTest2(1, T) \
	genTest2(2, T) \
	genTest2(3, T) \
	genTest2(4, T) \
	genTest2(5, T)

		genTest3(uint8_t)
		genTest3(uint16_t)
		genTest3(uint32_t)
		genTest3(uint64_t)
		genTest3(int8_t)
		genTest3(int16_t)
		genTest3(int32_t)
		genTest3(int64_t)
		genTest3(float)
		genTest3(double)
	};
}