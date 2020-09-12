#pragma once

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