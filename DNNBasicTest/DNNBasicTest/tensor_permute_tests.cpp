#include <CppUnitTest.h>
#include <string>
#include "tensor.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorPermuteTests)
	{
	public:
		template<typename T>
		void matrixPermute2x4()
		{
			dnnbasic::tensor<T> input({ 4, 2 },
				{
					5,4,
					6,7,
					8,1,
					0,0
				});

			dnnbasic::tensor<T> expected({ 2, 4 },
				{
					5,6,8,0,
					4,7,1,0
				});

			auto* actual = input.permute({ 1,0 });
			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixPermute2x4)

			template<typename T>
		void matrixPermute4x4()
		{
			dnnbasic::tensor<T> input({ 4, 4 },
				{
					1,0,1,0,
					0,1,0,0,
					0,0,1,0,
					0,0,0,1
				});

			dnnbasic::tensor<T> expected({ 4, 4 },
				{
					1,0,0,0,
					0,1,0,0,
					1,0,1,0,
					0,0,0,1
				});

			auto* actual = input.permute({ 1,0 });
			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixPermute4x4)

		template<typename T>
		void matrixPermute2x2x4()
		{
			dnnbasic::tensor<T> input({ 2,2,4 },
				{
					0,1,2,3,
					4,5,6,7,

					8,9,10,11,
					12,13,14,15
				});

			dnnbasic::tensor<T> expected({ 2,2,4 },
				{
					0,1,2,3,
					8,9,10,11,

					4,5,6,7,
					12,13,14,15
				});

			auto* actual = input.permute({ 1,0,2 });

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixPermute2x2x4)

			template<typename T>
		void matrixPermute2x4NamedDims()
		{
			dnnbasic::tensor<T> input({ 4, 2 }, { "height", "width" },
				{
					5,4,
					6,7,
					8,1,
					0,0
				});

			dnnbasic::tensor<T> expected({ 2, 4 },
				{
					5,6,8,0,
					4,7,1,0
				});

			auto* actual = input.permute({ "width", "height" });
			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixPermute2x4NamedDims)

			template<typename T>
		void matrixPermute4x4NamedDims()
		{
			dnnbasic::tensor<T> input({ 4, 4 }, { "height", "width" },
				{
					1,0,1,0,
					0,1,0,0,
					0,0,1,0,
					0,0,0,1
				});

			dnnbasic::tensor<T> expected({ 4, 4 },
				{
					1,0,0,0,
					0,1,0,0,
					1,0,1,0,
					0,0,0,1
				});

			auto* actual = input.permute({ "width", "height" });
			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixPermute4x4NamedDims)
	};
}