#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <chrono>
#include "cudaBasics.h"
#include "tensor.h"
#include "linear.h"
#include "mean_squared_loss.h"
#include "sgd.h"
#include "graphRecorder.h"
#include "mnist.h"
#include "relu.h"
#include "cross_entropy_loss.h"

void printSample(int32_t index, dnnbasic::tensor<uint8_t> input, dnnbasic::tensor<float> actual)
{
	std::vector<uint8_t> inVals = input.getValuesOnCPU();
	std::vector<float> actVals = actual.getValuesOnCPU();

	for (size_t y = 0; y < 28; y++)
	{
		for (size_t x = 0; x < 28; x++)
		{
			uint8_t val = inVals[index * 28 * 28 + y * 28 + x];
			if (val < 50)
			{
				std::cout << ".";
			}
			else if (val < 200)
			{
				std::cout << "+";
			}
			else
			{
				std::cout << "#";
			}
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	size_t bestIndex = 0;
	float bestValue = -10000;
	for (size_t i = 0; i < 10; i++)
	{
		if (actVals[index * 10 + i] > bestValue)
		{
			bestValue = actVals[index * 10 + i];
			bestIndex = i;
		}
	}

	std::cout << bestIndex << std::endl;
	std::cout << std::endl;
}

int main()
{
	{
		const uint32_t batchSize = 100;
		auto dataset = dnnbasic::datasets::mnist::loadTrainingSet("C:/Users/Peter/Desktop/mnist", batchSize);
		auto test_dataset = dnnbasic::datasets::mnist::loadTestSet("C:/Users/Peter/Desktop/mnist", batchSize);

		auto* opti = new dnnbasic::optimizer::sgd(0.00001f);

		dnnbasic::layer::linear<float> l1(28 * 28, 256, true);
		dnnbasic::layer::linear<float> l2(l1.getOutputSize(), 128, true);
		dnnbasic::layer::linear<float> l3(l2.getOutputSize(), 10, true);

		dnnbasic::activations::relu<float> dwa;

		dnnbasic::tensor<uint8_t> input({ batchSize, 28, 28 }, { "batch", "height", "width" });
		dnnbasic::tensor<uint8_t> output({ batchSize, 10 });

		dnnbasic::graphRecorder recorder;

		recorder.startRecording();

		auto x = input.cast<float>() / 255.0f;
		x = l1.forward(x.reshape("batch", l1.getInputSize()));
		x = dwa.forward(x);
		x = l2.forward(x);
		x = dwa.forward(x);
		x = l3.forward(x);

		auto y = output.cast<float>();
		auto fisk = dnnbasic::loss::crossEntropyLoss(y, x);
		fisk.backward(opti);

		recorder.stopRecording();


		for (size_t epoch = 0; epoch < 100; epoch++)
		{
			auto start = std::chrono::system_clock::now();

			for (size_t i = 0; i < dataset.getBatchCount(); i++)
			{

				auto [in, out] = dataset[i];
				in.copyTo(input);
				out.copyTo(output);

				recorder.replay();

				//auto x = input.cast<float>() / 255.0f;
				//x = l1.forward(x.reshape("batch", l1.getInputSize()));
				//x = dwa.forward(x);
				//x = l2.forward(x);
				//x = dwa.forward(x);
				//x = l3.forward(x);
				//////x = dwa.forward(x);
				////x = x.reshape("batch", 10);


				//auto y = output.cast<float>();
				//auto fisk = dnnbasic::loss::crossEntropyLoss(y, x);
				//fisk.backward(opti);

			}

			auto end = std::chrono::system_clock::now();

			float accuracy = 0.0f;
			float total = test_dataset.getBatchCount() * batchSize;

			for (size_t i = 0; i < test_dataset.getBatchCount(); i++)
			{
				auto [in, out] = test_dataset[i];
				in.copyTo(input);
				out.copyTo(output);

				recorder.replay();

				auto labels = output.getValuesOnCPU();
				auto predicted = x.getValuesOnCPU();

				for (size_t j = 0; j < batchSize; j++)
				{
					size_t bestIndex = 0;
					float bestValue = -10000;
					for (size_t k = 0; k < 10; k++)
					{
						if (predicted[j * 10 + k] > bestValue)
						{
							bestValue = predicted[j * 10 + k];
							bestIndex = k;
						}
					}
					if (labels[j * 10 + bestIndex] == 1)
					{
						accuracy += 1.0f;
						//std::cout << std::endl;
						//printSample(j, input, x);
					}
					
				}
			}

			std::cout << "Accuracy: " << (accuracy / total) * 100 << std::endl;

			
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

			std::cout << "Error: " << fisk.getError() << std::endl;
			//printSample(epoch % batchSize, input, x);
			std::cout << "Time: " << time.count() << std::endl;
		}
	}

	cudabasic::resetCudaDevice();
}