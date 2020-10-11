#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <random>

namespace dnnbasic::random
{
	void setRandomSeed(const uint32_t seed);

	template<typename T>
	std::vector<T> getRandomNumbers(const uint32_t size, const T min, const T max);
}