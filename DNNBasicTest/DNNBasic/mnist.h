#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include "dataloader.h"
#include "tensor.h"
#include "supervisedDataset.h"



namespace dnnbasic::datasets::mnist
{
	supervisedDataset<uint8_t> loadTrainingSet(
		const std::string& folderPath, 
		const uint32_t batchSize,
		const std::string& batchName = "batch",
		const std::string& heightName = "height",
		const std::string& widthName = "width",
		const std::string& labelName = "label");
	supervisedDataset<uint8_t> loadTestSet(
		const std::string& folderPath, 
		const uint32_t batchSize,
		const std::string& batchName = "batch",
		const std::string& heightName = "height",
		const std::string& widthName = "width",
		const std::string& labelName = "label");
}