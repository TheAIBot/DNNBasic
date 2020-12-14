#include <CppUnitTest.h>
#include <string>
#include "tensor.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorMaxTests)
	{
	public:
		template<typename T>
		void max2x2Dim0()
		{
			dnnbasic::tensor<T> a({ 2, 2 },
				{
					3, 4,
					5, 6
				});

			dnnbasic::tensor<T> expected({ 2 },
				{
					5, 6
				});

			assertTensorOp<T>(expected, [&]() {return a.max(0); });
		}
		TEST_UNSIGNED_OP_TYPES(max2x2Dim0)

			template<typename T>
		void max2x2Dim1()
		{
			dnnbasic::tensor<T> a({ 2, 2 },
				{
					3, 4,
					5, 6
				});

			dnnbasic::tensor<T> expected({ 2 },
				{
					4,
					6
				});

			assertTensorOp<T>(expected, [&]() {return a.max(1); });
		}
		TEST_UNSIGNED_OP_TYPES(max2x2Dim1)

			template<typename T>
		void max2x2x2Dim0()
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
					5, 7,
					5, 6
				});

			assertTensorOp<T>(expected, [&]() {return a.max(0); });
		}
		TEST_UNSIGNED_OP_TYPES(max2x2x2Dim0)

			template<typename T>
		void max2x2x2Dim1()
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
					5, 6,

					5, 7
				});

			assertTensorOp<T>(expected, [&]() {return a.max(1); });
		}
		TEST_UNSIGNED_OP_TYPES(max2x2x2Dim1)

			template<typename T>
		void max2x2x2Dim2()
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
					4,
					6,

					7,
					2
				});

			assertTensorOp<T>(expected, [&]() {return a.max(2); });
		}
		TEST_UNSIGNED_OP_TYPES(max2x2x2Dim2)


			template<typename T>
		void max3x1x3x2Dim0()
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
					6, 9,
					5, 6,
					8, 7
				});

			assertTensorOp<T>(expected, [&]() {return a.max(0); });
		}
		TEST_UNSIGNED_OP_TYPES(max3x1x3x2Dim0)

			template<typename T>
		void max3x1x3x2Dim1()
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

			assertTensorOp<T>(expected, [&]() {return a.max(1); });
		}
		TEST_UNSIGNED_OP_TYPES(max3x1x3x2Dim1)

			template<typename T>
		void max3x1x3x2Dim2()
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
					5, 7,

					8, 6,

					6, 9
				});

			assertTensorOp<T>(expected, [&]() {return a.max(2); });
		}
		TEST_UNSIGNED_OP_TYPES(max3x1x3x2Dim2)

			template<typename T>
		void max3x1x3x2Dim3()
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
					4,
					6,
					7,

					2,
					6,
					8,

					9,
					6,
					4
				});

			assertTensorOp<T>(expected, [&]() {return a.max(3); });
		}
		TEST_UNSIGNED_OP_TYPES(max3x1x3x2Dim3)
	};
}