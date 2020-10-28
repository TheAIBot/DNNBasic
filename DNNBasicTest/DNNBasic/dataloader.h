#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace dnnbasic::datasets
{
	class dataloader
	{
	protected:
		std::ifstream fileStream;

	public:
		dataloader(const std::string& filepath)
		{ 
			if (std::filesystem::exists(filepath))
			{
				fileStream = std::ifstream(filepath, std::ios_base::binary);
			}
			else
			{
				throw std::runtime_error("File does not exist.");
			}
		}

		std::vector<char> loadNext(const uint32_t size)
		{
			std::vector<char> buffer(size);
			fileStream.read(&buffer[0], size);

			return buffer;
		}
	};
}