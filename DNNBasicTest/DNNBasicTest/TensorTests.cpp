#include "CppUnitTest.h"
#include "Tensor.h"
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace Microsoft
{
	namespace VisualStudio
	{
		namespace CppUnitTestFramework
		{
			template<> 
			static std::wstring ToString<dnnbasic::tensor<float>>(const dnnbasic::tensor<float>& t)
			{ 
				return L"FloatTensor"; 
			}
			template<>
			static std::wstring ToString<const dnnbasic::tensor<float>&>(const dnnbasic::tensor<float>& t)
			{
				return L"FloatTensor";
			}
		}
	}
}

namespace DNNBasicTest
{
	TEST_CLASS(TensorTests)
	{
	public:
		
		TEST_METHOD(TensorMul)
		{
			dnnbasic::tensor<float> a({ 2, 1, 3 }, {1, 2, 3, 4, 5, 6});
			dnnbasic::tensor<float> b({ 2, 1, 3 }, {3, 4, 5, 6, 7, 8});

			dnnbasic::tensor<float> expected({ 2, 1, 3 }, { 3, 8, 15, 24, 35, 48 });
			auto* actual = a * b;

			Assert::AreEqual(expected, *actual);
		}

		TEST_METHOD(TensorAddTensor)
		{
			dnnbasic::tensor<float> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			dnnbasic::tensor<float> b({ 2, 1, 3 }, { 3, 4, 5, 6, 7, 8 });

			dnnbasic::tensor<float> expected({ 2, 1, 3 }, { 4,6,8,10,12,14 });
			auto* actual = a + b;

			Assert::AreEqual(expected, *actual);
		}
	};
}
