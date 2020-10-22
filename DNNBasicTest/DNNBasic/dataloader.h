#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>

namespace dnnbasic::datasets
{
	class dataloader
	{
	protected:
		std::ifstream fileStream;

	public:
		dataloader(const std::string& filepath) : fileStream(filepath, std::ios_base::binary)
		{ }

		std::vector<char> loadNext(const uint32_t size)
		{
			std::vector<char> buffer(size);
			fileStream.read(&buffer[0], size);

			return buffer;
		}
	};
}