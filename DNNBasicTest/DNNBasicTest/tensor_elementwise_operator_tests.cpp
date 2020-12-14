#include <CppUnitTest.h>
#include <string>
#include <functional>
#include <vector>
#include <array>
#include "tensor.h"
#include "test_tools.h"

namespace DNNBasicTest
{
	TEST_CLASS(tensorElementwiseOpTests)
	{
	public:
		template<typename T>
		void tensorMulTensor()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, {1, 2, 3, 4, 5, 6});
			dnnbasic::tensor<T> b({ 2, 1, 3 }, {3, 4, 5, 6, 7, 8});

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)3, (T)8, (T)15, (T)24, (T)35, (T)48 });
			auto actual = a * b;

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(tensorMulTensor)

		template<typename T>
		void TensorMulScalar()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			T b = 5;

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)5, (T)10, (T)15, (T)20, (T)25, (T)30 });
			auto actual = a * b;

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(TensorMulScalar)

		template<typename T>
		void ScalarMulTensor()
		{
			T a = 4;
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 3, 4, 5, 6, 7, 8 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)12, (T)16, (T)20, (T)24, (T)28, (T)32 });
			auto actual = a * b;

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(ScalarMulTensor)

		template<typename T>
		void TensorAddTensor()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 3, 4, 5, 6, 7, 8 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)4, (T)6, (T)8, (T)10, (T)12, (T)14 });
			auto actual = a + b;

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(TensorAddTensor)

		template<typename T>
		void TensorAddScalar()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			T b = 7;

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)8, (T)9, (T)10, (T)11, (T)12, (T)13 });
			auto actual = a + b;

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(TensorAddScalar)

		template<typename T>
		void ScalarAddTensor()
		{
			T a = 27;
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)28, (T)29, (T)30, (T)31, (T)32, (T)33 });
			auto actual = a + b;

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(ScalarAddTensor)

		template<typename T>
		void TensorSubTensor()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 3, 4, 5, 6, 7, 8 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)-2, (T)-2, (T)-2, (T)-2, (T)-2, (T)-2 });
			auto actual = a - b;

			Assert::AreEqual(expected, actual);
		}
		TEST_SIGNED_OP_TYPES(TensorSubTensor)

		template<typename T>
		void TensorSubScalar()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });
			T b = 13;

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)-12, (T)-11, (T)-10, (T)-9, (T)-8, (T)-7 });
			auto actual = a - b;

			Assert::AreEqual(expected, actual);
		}
		TEST_SIGNED_OP_TYPES(TensorSubScalar)

		template<typename T>
		void ScalarSubTensor()
		{
			T a = 47;
			dnnbasic::tensor<T> b({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)46, (T)45, (T)44, (T)43, (T)42, (T)41 });
			auto actual = a - b;

			Assert::AreEqual(expected, actual);
		}
		TEST_SIGNED_OP_TYPES(ScalarSubTensor)

		template<typename T>
		void TensorSub()
		{
			dnnbasic::tensor<T> a({ 2, 1, 3 }, { 1, 2, 3, 4, 5, 6 });

			dnnbasic::tensor<T> expected({ 2, 1, 3 }, { (T)-1, (T)-2, (T)-3, (T)-4, (T)-5, (T)-6});
			auto actual = -a;

			Assert::AreEqual(expected, actual);
		}
		TEST_SIGNED_OP_TYPES(TensorSub)

		template<typename T>
		void tensorAdd1x2_2x1()
		{
			dnnbasic::tensor<T> a({ 1, 2 },
				{
					2, 3
				});
			dnnbasic::tensor<T> b({ 2, 1 },
				{
					5,
					6
				});

			dnnbasic::tensor<T> expected({ 2, 2 },
				{
					7, 8,
					8, 9
				});
			auto actual = a + b;

			Assert::AreEqual(expected, actual);
		}
		TEST_ALL_OP_TYPES(tensorAdd1x2_2x1)
	};

	TEST_CLASS(tensorElementwiseOpTestsAutoGenerated)
	{
		std::array<std::vector<uint32_t>, 4> testDims =
		{
			std::vector<uint32_t>({ 6234 }),
			std::vector<uint32_t>({ 12, 353, 2}),
			std::vector<uint32_t>({ 3, 5, 2, 56, 1}),
			std::vector<uint32_t>({ 424, 42, 5})
		};
		std::array<int32_t, 4> testScalars = { -5, 1, 5, 37 };
	public:
		template<typename T>
		void elementWiseTensorTensor(std::vector<uint32_t> dims, std::function<dnnbasic::tensor<T>(dnnbasic::tensor<T>, dnnbasic::tensor<T>)> tenFunc, std::function<T(T, T)> typFunc)
		{
			dnnbasic::tensor<T> a = dnnbasic::tensor<T>::random(dims);
			dnnbasic::tensor<T> b = dnnbasic::tensor<T>::random(dims);

			std::vector<T> aData = a.getValuesOnCPU();
			std::vector<T> bData = b.getValuesOnCPU();

			std::vector<T> expectedData;
			for (size_t i = 0; i < aData.size(); i++)
			{
				expectedData.push_back(typFunc(aData[i], bData[i]));
			}

			dnnbasic::tensor<T> expected(dims, expectedData);

			assertCloseEnoughTensorOp<T>(expected, [&]() {return tenFunc(a, b); });
		}

		template<typename T>
		void elementWiseTensorScalar(std::vector<uint32_t> dims, T scalar, std::function<dnnbasic::tensor<T>(dnnbasic::tensor<T>, T)> tenFunc, std::function<T(T, T)> typFunc)
		{
			dnnbasic::tensor<T> a = dnnbasic::tensor<T>::random(dims);

			std::vector<T> aData = a.getValuesOnCPU();

			std::vector<T> expectedData;
			for (size_t i = 0; i < aData.size(); i++)
			{
				expectedData.push_back(typFunc(aData[i], scalar));
			}

			dnnbasic::tensor<T> expected(dims, expectedData);

			assertCloseEnoughTensorOp<T>(expected, [&]() {return tenFunc(a, scalar); });
		}

		template<typename T>
		void elementWiseScalarTensor(std::vector<uint32_t> dims, T scalar, std::function<dnnbasic::tensor<T>(T, dnnbasic::tensor<T>)> tenFunc, std::function<T(T, T)> typFunc)
		{
			dnnbasic::tensor<T> b = dnnbasic::tensor<T>::random(dims);

			std::vector<T> bData = b.getValuesOnCPU();

			std::vector<T> expectedData;
			for (size_t i = 0; i < bData.size(); i++)
			{
				expectedData.push_back(typFunc(scalar, bData[i]));
			}

			dnnbasic::tensor<T> expected(dims, expectedData);

			assertCloseEnoughTensorOp<T>(expected, [&]() {return tenFunc(scalar, b); });
		}

#define GEN_TEST_TT_TYPE_DIMS_FUNC(type, dimsIdx, func, funcStr) \
	TEST_METHOD(type ## elementwiseTensorTensor ## _ ## dimsIdx ## _ ## funcStr) { elementWiseTensorTensor<type>(testDims[dimsIdx], [](auto a, auto b) { return a func b;}, [](auto a, auto b) { return a func b;}); }

#define GEN_TEST_TS_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, scalarIdx, func, funcStr) \
	TEST_METHOD(type ## elementWiseTensorScalar ## _ ## dimsIdx ## _ ## scalarIdx ## _ ## funcStr) { elementWiseTensorScalar<type>(testDims[dimsIdx], (type)testScalars[scalarIdx], [](auto a, auto b) { return a func b; }, [](auto a, auto b) { return a func b; }); }

#define GEN_TEST_ST_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, scalarIdx, func, funcStr) \
	TEST_METHOD(type ## elementWiseScalarTensor ## _ ## dimsIdx ## _ ## scalarIdx ## _ ## funcStr) { elementWiseScalarTensor<type>(testDims[dimsIdx], (type)testScalars[scalarIdx], [](auto a, auto b) { return a func b; }, [](auto a, auto b) { return a func b; }); }

#define GEN_TEST_TS_TYPE_DIMS_FUNC(type, dimsIdx, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 0, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 1, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 2, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 3, func, funcStr)

#define GEN_TEST_ST_TYPE_DIMS_FUNC(type, dimsIdx, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 0, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 1, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 2, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_SCALAR_FUNC(type, dimsIdx, 3, func, funcStr)

#define GEN_TEST_TT_TYPE_FUNC(type, func, funcStr) \
	GEN_TEST_TT_TYPE_DIMS_FUNC(type, 0, func, funcStr) \
	GEN_TEST_TT_TYPE_DIMS_FUNC(type, 1, func, funcStr) \
	GEN_TEST_TT_TYPE_DIMS_FUNC(type, 2, func, funcStr) \
	GEN_TEST_TT_TYPE_DIMS_FUNC(type, 3, func, funcStr)

#define GEN_TEST_TS_TYPE_FUNC(type, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_FUNC(type, 0, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_FUNC(type, 1, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_FUNC(type, 2, func, funcStr) \
	GEN_TEST_TS_TYPE_DIMS_FUNC(type, 3, func, funcStr)

#define GEN_TEST_ST_TYPE_FUNC(type, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_FUNC(type, 0, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_FUNC(type, 1, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_FUNC(type, 2, func, funcStr) \
	GEN_TEST_ST_TYPE_DIMS_FUNC(type, 3, func, funcStr)

#define GEN_TEST_TTTSST_TYPE_FUNC(type, func, funcStr) \
	GEN_TEST_TT_TYPE_FUNC(type, func, funcStr) \
	GEN_TEST_TS_TYPE_FUNC(type, func, funcStr) \
	GEN_TEST_ST_TYPE_FUNC(type, func, funcStr)

		GEN_TEST_TTTSST_TYPE_FUNC(uint8_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(uint16_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(uint32_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(uint64_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(int8_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(int16_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(int32_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(int64_t, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(float, +, plus)
		GEN_TEST_TTTSST_TYPE_FUNC(double, +, plus)

		GEN_TEST_TTTSST_TYPE_FUNC(int8_t, -, sub)
		GEN_TEST_TTTSST_TYPE_FUNC(int16_t, -, sub)
		GEN_TEST_TTTSST_TYPE_FUNC(int32_t, -, sub)
		GEN_TEST_TTTSST_TYPE_FUNC(int64_t, -, sub)
		GEN_TEST_TTTSST_TYPE_FUNC(float, -, sub)
		GEN_TEST_TTTSST_TYPE_FUNC(double, -, sub)

		GEN_TEST_TTTSST_TYPE_FUNC(uint8_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(uint16_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(uint32_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(uint64_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(int8_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(int16_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(int32_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(int64_t, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(float, *, mul)
		GEN_TEST_TTTSST_TYPE_FUNC(double, *, mul)

		//GEN_TEST_TTTSST_TYPE_FUNC(uint8_t, /, div)
		//GEN_TEST_TTTSST_TYPE_FUNC(uint16_t, /, div)
		//GEN_TEST_TTTSST_TYPE_FUNC(uint32_t, /, div)
		//GEN_TEST_TTTSST_TYPE_FUNC(uint64_t, /, div)
		//GEN_TEST_TTTSST_TYPE_FUNC(int8_t, /, div)
		//GEN_TEST_TTTSST_TYPE_FUNC(int16_t, /, div)
		//GEN_TEST_TTTSST_TYPE_FUNC(int32_t, /, div)
		//GEN_TEST_TTTSST_TYPE_FUNC(int64_t, /, div)
		GEN_TEST_TTTSST_TYPE_FUNC(float, /, div)
		GEN_TEST_TTTSST_TYPE_FUNC(double, /, div)
	};
}
