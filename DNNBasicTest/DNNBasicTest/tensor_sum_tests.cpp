#include <CppUnitTest.h>
#include <string>
#include "tensor.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorSumTests)
	{
	public:
		template<typename T>
		void sum2x2SumDim0()
		{
			dnnbasic::tensor<T> a({ 2, 2 },
				{
					3, 4,
					5, 6
				});

			dnnbasic::tensor<T> expected({ 2 },
				{
					8, 10
				});

			auto actual = a.sum(0);
			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(sum2x2SumDim0)

		template<typename T>
		void sum2x2SumDim1()
		{
			dnnbasic::tensor<T> a({ 2, 2 },
				{
					3, 4,
					5, 6
				});

			dnnbasic::tensor<T> expected({ 2 },
				{
					7,
					11
				});

			auto actual = a.sum(1);
			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(sum2x2SumDim1)

		template<typename T>
		void sum2x2x2SumDim0()
		{
			dnnbasic::tensor<T> a({ 2, 2, 2 },
				{
					3, 4,
					5, 6,

					5, 7,
					2, 1
				});

			dnnbasic::tensor<T> expected({ 2, 2 },
				{
					8, 11,
					7, 7
				});

			auto actual = a.sum(0);
			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(sum2x2x2SumDim0)

		template<typename T>
		void sum2x2x2SumDim1()
		{
			dnnbasic::tensor<T> a({ 2, 2, 2 },
				{
					3, 4,
					5, 6,

					5, 7,
					2, 1
				});

			dnnbasic::tensor<T> expected({ 2, 2 },
				{
					8, 10,

					7, 8
				});

			auto actual = a.sum(1);
			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(sum2x2x2SumDim1)

		template<typename T>
		void sum2x2x2SumDim2()
		{
			dnnbasic::tensor<T> a({ 2, 2, 2 },
				{
					3, 4,
					5, 6,

					5, 7,
					2, 1
				});

			dnnbasic::tensor<T> expected({ 2, 2 },
				{
					7,
					11,

					12,
					3
				});

			auto actual = a.sum(2);
			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(sum2x2x2SumDim2)
	};
}