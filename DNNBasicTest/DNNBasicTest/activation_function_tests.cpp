#include <CppUnitTest.h>
#include <string>
#include "tensor.h"
#include "relu.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorActivationTests)
	{
	public:
		template<typename T>
		void relu2x2()
		{
			dnnbasic::activations::relu<T> dwa;

			dnnbasic::tensor<T> actual({ 2, 2 },
				{
					3, -4,
					-5, 6
				});

			dnnbasic::tensor<T> expected({ 2, 2 },
				{
					3, 0,
					0, 6
				});

			assertTensorOp<T>(expected, [&]() {return dwa.forward(actual); });
		}
		TEST_SIGNED_OP_TYPES(relu2x2)

			template<typename T>
		void relu2x2random()
		{
			dnnbasic::activations::relu<T> dwa;

			dnnbasic::tensor<T> input = dnnbasic::tensor<T>::random({ 1000,43 });
			
			auto actual = dwa.forward(input);
			
			auto actualCPU = actual.getValuesOnCPU();

			for (size_t i = 0; i < actualCPU.size(); i++)
			{
				Assert::IsTrue(0 <= actualCPU[i]);
			}

		}
		TEST_SIGNED_OP_TYPES(relu2x2random)

			template<typename T>
		void relurelu2x2random()
		{
			dnnbasic::activations::relu<T> dwa;

			dnnbasic::tensor<T> input = dnnbasic::tensor<T>::random({ 1000,43 });

			auto actual = dwa.forward(input);

			auto expected = dwa.forward(actual);

			Assert::AreEqual(expected, actual);

		}
		TEST_SIGNED_OP_TYPES(relurelu2x2random)

	};
}