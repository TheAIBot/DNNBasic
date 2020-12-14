#pragma once

#include <CppUnitTest.h>
#include <typeinfo>
#include <codecvt>
#include <string>
#include <vector>
#include <random>
#include <functional>

#include "tensor.h"
#include "graphRecorder.h"

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

#define TEST_ALL_OP_TYPES_LEAST_8_BITS(methodName) \
	TEST_METHOD(uint16_t ## methodName) { methodName<uint16_t>(); } \
	TEST_METHOD(uint32_t ## methodName) { methodName<uint32_t>(); } \
	TEST_METHOD(uint64_t ## methodName) { methodName<uint64_t>(); } \
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

#define TEST_UNSIGNED_OP_TYPES(methodName) \
	TEST_METHOD(uint32_t ## methodName) { methodName<uint32_t>(); } \
	TEST_METHOD(uint64_t ## methodName) { methodName<uint64_t>(); } 

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

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
	template<typename T>
	std::vector<T> GetVectorWithRandomNumbers(const uint32_t size)
	{
		std::vector<T> numbers;

		std::default_random_engine rngGen(size);
		if constexpr (std::is_floating_point<T>::value)
		{
			std::uniform_real_distribution<T> dist(-13722, 64323);
			for (size_t i = 0; i < size; i++)
			{
				numbers.push_back(dist(rngGen));
			}
		}
		else if constexpr (std::is_signed<T>::value)
		{
			std::uniform_int_distribution<int32_t> dist(-13722, 64323);
			for (size_t i = 0; i < size; i++)
			{
				numbers.push_back((T)dist(rngGen));
			}
		}
		else if constexpr (std::is_unsigned<T>::value)
		{
			std::uniform_int_distribution<uint32_t> dist(0, 64323);
			for (size_t i = 0; i < size; i++)
			{
				numbers.push_back((T)dist(rngGen));
			}
		}
		else
		{
			static_assert("Failed to make a random generator for the specified type.");
		}

		return numbers;
	}

	template<typename T>
	void resultCloseEnough(dnnbasic::tensor<T> expected, dnnbasic::tensor<T> actual)
	{
		auto expVals = expected.getValuesOnCPU();
		auto actVals = actual.getValuesOnCPU();

		if constexpr (std::is_floating_point<T>::value)
		{
			Assert::AreEqual(expVals.size(), actVals.size());
			for (size_t i = 0; i < expVals.size(); i++)
			{
				if (std::abs(expVals[i] - actVals[i]) >= (T)0.0001)
				{
					Assert::Fail();
				}
			}
		}
		else
		{
			Assert::IsTrue(expVals == actVals);
		}
	}

	template<typename T>
	void assertTensorOp(const dnnbasic::tensor<T> expected, std::function<dnnbasic::tensor<T>()> tensorOp)
	{
		//op gives correct result
		auto withoutRecordingFirst = tensorOp();
		Assert::AreEqual(expected, withoutRecordingFirst);

		//running op twice should still gave the same result
		auto withoutRecordingSecond = tensorOp();
		Assert::AreEqual(withoutRecordingFirst, withoutRecordingSecond);

		dnnbasic::graphRecorder recorder;
		recorder.startRecording();

		auto withRecording = tensorOp();

		recorder.stopRecording();

		//replaying should give same result
		recorder.replay();
		auto withRecordingFirst = withRecording + (T)0;
		Assert::AreEqual(expected, withRecordingFirst);

		//replaying twice should still give same result
		recorder.replay();
		auto withRecordingSecond = withRecording + (T)0;
		Assert::AreEqual(withRecordingFirst, withRecordingSecond);
	}

	template<typename T>
	void assertCloseEnoughTensorOp(const dnnbasic::tensor<T> expected, std::function<dnnbasic::tensor<T>()> tensorOp)
	{
		auto withoutRecordingFirst = tensorOp();

		//op gives correct result
		resultCloseEnough(expected, withoutRecordingFirst);

		auto withoutRecordingSecond = tensorOp();

		//running op twice should still gave the same result
		resultCloseEnough(withoutRecordingFirst, withoutRecordingSecond);

		dnnbasic::graphRecorder recorder;
		recorder.startRecording();

		auto withRecording = tensorOp();

		recorder.stopRecording();

		//replaying should give same result
		recorder.replay();
		auto withRecordingFirst = withRecording + (T)0;
		resultCloseEnough(expected, withRecordingFirst);

		//replaying twice should still give same result
		recorder.replay();
		auto withRecordingSecond = withRecording + (T)0;
		resultCloseEnough(withRecordingFirst, withRecordingSecond);
	}
}