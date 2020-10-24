#include <thread>
#include "random.h"


namespace dnnbasic::random
{
	static thread_local uint32_t rngSeed = 27;
	static thread_local std::default_random_engine rngGen(rngSeed);

	void setRandomSeed(const uint32_t seed)
	{
		rngSeed = seed;
	}

	template<typename T>
	std::vector<T> getRandomNumbers(const uint32_t size, const T min, const T max)
	{
		std::vector<T> numbers;

		if constexpr (std::is_floating_point<T>::value)
		{
			std::uniform_real_distribution<T> dist(min, max);
			for (size_t i = 0; i < size; i++)
			{
				numbers.push_back(dist(rngGen));
			}
		}
		else if constexpr (std::is_signed<T>::value)
		{
			std::uniform_int_distribution<int64_t> dist(min, max);
			for (size_t i = 0; i < size; i++)
			{
				numbers.push_back((T)dist(rngGen));
			}
		}
		else if constexpr (std::is_unsigned<T>::value)
		{
			std::uniform_int_distribution<uint64_t> dist(min, max);
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

	template std::vector<bool> getRandomNumbers(const uint32_t size, const bool min, const bool max);
	template std::vector<uint8_t> getRandomNumbers(const uint32_t size, const uint8_t min, const uint8_t max);
	template std::vector<uint16_t> getRandomNumbers(const uint32_t size, const uint16_t min, const uint16_t max);
	template std::vector<uint32_t> getRandomNumbers(const uint32_t size, const uint32_t min, const uint32_t max);
	template std::vector<uint64_t> getRandomNumbers(const uint32_t size, const uint64_t min, const uint64_t max);
	template std::vector<int8_t> getRandomNumbers(const uint32_t size, const int8_t min, const int8_t max);
	template std::vector<int16_t> getRandomNumbers(const uint32_t size, const int16_t min, const int16_t max);
	template std::vector<int32_t> getRandomNumbers(const uint32_t size, const int32_t min, const int32_t max);
	template std::vector<int64_t> getRandomNumbers(const uint32_t size, const int64_t min, const int64_t max);
	template std::vector<float> getRandomNumbers(const uint32_t size, const float min, const float max);
	template std::vector<double> getRandomNumbers(const uint32_t size, const double min, const double max);
}
