#include <CppUnitTest.h>
#include <string>
#include <typeinfo>
#include <codecvt>
#include "Tensor.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#define TEST_ALL_OP_TYPES(methodName) \
	TEST_METHOD(uint8_t ## methodName) { methodName<uint8_t>(); } \
	TEST_METHOD(uint16_t ## methodName) { methodName<uint16_t>(); } \
	TEST_METHOD(uint32_t ## methodName) { methodName<uint32_t>(); } \
	TEST_METHOD(uint64_t ## methodName) { methodName<uint64_t>(); } \
	TEST_METHOD(int8_t ## methodName) { methodName<int8_t>(); } \
	TEST_METHOD(int16_t ## methodName) { methodName<int16_t>(); } \
	TEST_METHOD(int32_t ## methodName) { methodName<int32_t>(); } \
	TEST_METHOD(int64_t ## methodName) { methodName<int64_t>(); } \
	TEST_METHOD(float ## methodName) { methodName<float>(); } \
	TEST_METHOD(double ## methodName) { methodName<double>(); }

#define TEST_SIGNED_OP_TYPES(methodName) \
	TEST_METHOD(int8_t ## methodName) { methodName<int8_t>(); } \
	TEST_METHOD(int16_t ## methodName) { methodName<int16_t>(); } \
	TEST_METHOD(int32_t ## methodName) { methodName<int32_t>(); } \
	TEST_METHOD(int64_t ## methodName) { methodName<int64_t>(); } \
	TEST_METHOD(float ## methodName) { methodName<float>(); } \
	TEST_METHOD(double ## methodName) { methodName<double>(); }

namespace Microsoft::VisualStudio::CppUnitTestFramework
{
	template<typename T>
	static std::wstring ToString(const dnnbasic::tensor<T>& t)
	{
		//get type name as a string
		std::string typeName = typeid(T).name();

		//convert typeName to wstring
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring typeNameW = converter.from_bytes(typeName);

		return typeNameW + L" Tensor";
	}
}

namespace DNNBasicTest
{
	TEST_CLASS(TensorTests)
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

			auto* actual = dnnbasic::permute(input, { 1,0 });

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixPermute2x4)

		template<typename T>
		void matrixPermute4x4()
		{
			dnnbasic::tensor<T> input({ 4, 4 },
				{
					1,0,0,0,
					0,1,0,0,
					0,0,1,0,
					0,0,0,1
				});

			dnnbasic::tensor<T> expected({ 4, 4 },
				{
					1,0,0,0,
					0,1,0,0,
					0,0,1,0,
					0,0,0,1
				});

			auto* actual = dnnbasic::permute(input, { 1,0 });

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixPermute4x4)
		template<typename T>
		void matrixMatrixProduct2x2() 
		{
			dnnbasic::tensor<T> a({ 2, 2 }, { 1,0,0,1 });
			dnnbasic::tensor<T> b({ 2, 2 }, { 4,1,1,2 });

			dnnbasic::tensor<T> expected({ 2, 2 }, { 4,1,1,2 });
			auto* actual = dnnbasic::matMul(a, b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixMatrixProduct2x2)

		template<typename T>
		void matrixMatrixProduct3x3()
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
			auto* actual = dnnbasic::matMul(a, b);

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(matrixMatrixProduct3x3)

		template<typename T>
		void tensorMulTensor()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, {1, 2, 3, 4, 5, 6});
			dnnbasic::tensor<T> b({ 2, 1, 3 }, {3, 4, 5, 6, 7, 8});

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)3, (T)8, (T)15, (T)24, (T)35, (T)48 });
			auto* actual = a * b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(tensorMulTensor)

		template<typename T>
		void TensorMulScalar()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			T b = 5;

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)5, (T)10, (T)15, (T)20, (T)25, (T)30 });
			auto* actual = a * b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(TensorMulScalar)

		template<typename T>
		void ScalarMulTensor()
		{
			T a = 4;
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 3, 4, 5, 6, 7, 8 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)12, (T)16, (T)20, (T)24, (T)28, (T)32 });
			auto* actual = a * b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(ScalarMulTensor)

		template<typename T>
		void TensorAddTensor()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 3, 4, 5, 6, 7, 8 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)4, (T)6, (T)8, (T)10, (T)12, (T)14 });
			auto* actual = a + b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(TensorAddTensor)

		template<typename T>
		void TensorAddScalar()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			T b = 7;

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)8, (T)9, (T)10, (T)11, (T)12, (T)13 });
			auto* actual = a + b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(TensorAddScalar)

		template<typename T>
		void ScalarAddTensor()
		{
			T a = 27;
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)28, (T)29, (T)30, (T)31, (T)32, (T)33 });
			auto* actual = a + b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_ALL_OP_TYPES(ScalarAddTensor)

		template<typename T>
		void TensorSubTensor()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 3, 4, 5, 6, 7, 8 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)-2, (T)-2, (T)-2, (T)-2, (T)-2, (T)-2 });
			auto* actual = a - b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_SIGNED_OP_TYPES(TensorSubTensor)

		template<typename T>
		void TensorSubScalar()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			T b = 13;

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)-12, (T)-11, (T)-10, (T)-9, (T)-8, (T)-7 });
			auto* actual = a - b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_SIGNED_OP_TYPES(TensorSubScalar)

		template<typename T>
		void ScalarSubTensor()
		{
			T a = 47;
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)46, (T)45, (T)44, (T)43, (T)42, (T)41 });
			auto* actual = a - b;

			Assert::AreEqual(expected, *actual);
		}
		TEST_SIGNED_OP_TYPES(ScalarSubTensor)

		template<typename T>
		void TensorSub()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)-1, (T)-2, (T)-3, (T)-4, (T)-5, (T)-6});
			auto* actual = -a;

			Assert::AreEqual(expected, *actual);
		}
		TEST_SIGNED_OP_TYPES(TensorSub)
	};
}
