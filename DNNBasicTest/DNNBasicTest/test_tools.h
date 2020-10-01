#pragma once

#include <typeinfo>
#include <codecvt>
#include <vector>
#include <random>

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

		std::default_random_engine rngGen(7);
		if constexpr (std::is_floating_point<T>::value)
		{
			std::uniform_real_distribution<T> dist(-1322, 64323);
			for (size_t i = 0; i < size; i++)
			{
				numbers.push_back(dist(rngGen));
			}
		}
		else if constexpr (std::is_signed<T>::value)
		{
			std::uniform_int_distribution<int32_t> dist(-1322, 64323);
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
}