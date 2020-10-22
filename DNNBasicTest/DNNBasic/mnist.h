#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include "dataloader.h"
#include "tensor.h"
#include "supervisedDataset.h"



namespace dnnbasic::datasets::mnist
{
	supervisedDataset<uint8_t> loadTrainingSet(const std::string& folderPath, const uint32_t batchSize);
	supervisedDataset<uint8_t> loadTestSet(const std::string& folderPath, const uint32_t batchSize);
}