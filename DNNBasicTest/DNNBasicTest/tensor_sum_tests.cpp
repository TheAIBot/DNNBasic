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

			assertTensorOp<T>(expected, [&]() {return a.sum(0); });
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

			assertTensorOp<T>(expected, [&]() {return a.sum(1); });
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

			assertTensorOp<T>(expected, [&]() {return a.sum(0); });
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

			assertTensorOp<T>(expected, [&]() {return a.sum(1); });
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

			assertTensorOp<T>(expected, [&]() {return a.sum(2); });
		}
		TEST_ALL_OP_TYPES(sum2x2x2SumDim2)

		template<typename T>
		void sum3x1x3x2SumDim0()
		{
			dnnbasic::tensor<T> a({ 3, 1, 3, 2 },
				{
					3, 4,
					5, 6,
					5, 7,

					2, 1,
					5, 6,
					8, 3,

					6, 9,
					3, 6,
					2, 4
				});

			dnnbasic::tensor<T> expected({ 1, 3, 2 },
				{
					11, 14,
					13, 18,
					15, 14
				});

			assertTensorOp<T>(expected, [&]() {return a.sum(0); });
		}
		TEST_ALL_OP_TYPES(sum3x1x3x2SumDim0)

		template<typename T>
		void sum3x1x3x2SumDim1()
		{
			dnnbasic::tensor<T> a({ 3, 1, 3, 2 },
				{
					3, 4,
					5, 6,
					5, 7,

					2, 1,
					5, 6,
					8, 3,

					6, 9,
					3, 6,
					2, 4
				});

			dnnbasic::tensor<T> expected({ 3, 3, 2 },
				{
					3, 4,
					5, 6,
					5, 7,

					2, 1,
					5, 6,
					8, 3,

					6, 9,
					3, 6,
					2, 4
				});

			assertTensorOp<T>(expected, [&]() {return a.sum(1); });
		}
		TEST_ALL_OP_TYPES(sum3x1x3x2SumDim1)

		template<typename T>
		void sum3x1x3x2SumDim2()
		{
			dnnbasic::tensor<T> a({ 3, 1, 3, 2 },
				{
					3, 4,
					5, 6,
					5, 7,

					2, 1,
					5, 6,
					8, 3,

					6, 9,
					3, 6,
					2, 4
				});

			dnnbasic::tensor<T> expected({ 3, 1, 2 },
				{
					13, 17,

					15, 10,

					11, 19
				});

			assertTensorOp<T>(expected, [&]() {return a.sum(2); });
		}
		TEST_ALL_OP_TYPES(sum3x1x3x2SumDim2)

		template<typename T>
		void sum3x1x3x2SumDim3()
		{
			dnnbasic::tensor<T> a({ 3, 1, 3, 2 },
				{
					3, 4,
					5, 6,
					5, 7,

					2, 1,
					5, 6,
					8, 3,

					6, 9,
					3, 6,
					2, 4
				});

			dnnbasic::tensor<T> expected({ 3, 1, 3 },
				{
					7,
					11,
					12,

					3,
					11,
					11,

					15,
					9,
					6
				});

			assertTensorOp<T>(expected, [&]() {return a.sum(3); });
		}
		TEST_ALL_OP_TYPES(sum3x1x3x2SumDim3)
	};
}