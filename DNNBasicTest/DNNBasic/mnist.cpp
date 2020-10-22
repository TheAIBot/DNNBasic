#include "mnist.h"

namespace dnnbasic::datasets::mnist
{
	static const std::string trainInputFile = "train-images-idx3-ubyte";
	static const std::string trainOutputFile = "train-labels-idx1-ubyte";
	static const std::string testInputFile = "t10k-images-idx3-ubyte";
	static const std::string testOutputFile = "t10k-labels-idx1-ubyte";

	static constexpr uint32_t inputHeaderSize = 4 * sizeof(uint32_t);
	static constexpr uint32_t outputHeaderSize = 2 * sizeof(uint32_t);

	static constexpr uint32_t trainSamplesCount = 60'000;
	static constexpr uint32_t testSamplesCount = 10'000;

	supervisedDataset<uint8_t> loadSet(const std::string& folderPath, const std::string& inputFileName, const std::string& outputFileName, const uint32_t sampleCount, const uint32_t batchSize)
	{
		dataloader input(folderPath + "/" + inputFileName);
		dataloader output(folderPath + "/" + outputFileName);

		//ignore headers
		input.loadNext(inputHeaderSize);
		input.loadNext(outputHeaderSize);

		std::vector<tensor<uint8_t>> inputs;
		std::vector<tensor<uint8_t>> outputs;

		const uint32_t batches = sampleCount / batchSize;
		for (size_t i = 0; i < batches; i++)
		{
			auto inputCharData = input.loadNext(batchSize * 28 * 28);
			auto outputCharData = input.loadNext(batchSize);

			std::vector<uint8_t> inputData(inputCharData.begin(), inputCharData.end());
			std::vector<uint8_t> outputData((size_t)batchSize * 10);
			for (size_t i = 0; i < batchSize; i++)
			{
				outputData[i * 10 + outputCharData[i]] = 1;
			}

			tensor<uint8_t> input({ batchSize, 1, 28, 28 }, inputData);
			tensor<uint8_t> output({ batchSize, 10 }, outputData);

			inputs.push_back(input);
			outputs.push_back(output);
		}

		return supervisedDataset<uint8_t>(inputs, outputs);
	}

	supervisedDataset<uint8_t> loadTrainingSet(const std::string& folderPath, const uint32_t batchSize)
	{
		return loadSet(folderPath, trainInputFile, trainOutputFile, trainSamplesCount, batchSize);
	}

	supervisedDataset<uint8_t> loadTestSet(const std::string& folderPath, const uint32_t batchSize)
	{
		return loadSet(folderPath, testInputFile, testOutputFile, testSamplesCount, batchSize);
	}
}