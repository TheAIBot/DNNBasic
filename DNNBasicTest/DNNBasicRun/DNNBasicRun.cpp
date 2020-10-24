// DNNBasicRun.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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

template<typename T>
struct supervisedTestData
{
	dnnbasic::tensor<T> input;
	dnnbasic::tensor<T> output;

	supervisedTestData(dnnbasic::tensor<T>& in, dnnbasic::tensor<T> out) : input(in), output(out)
	{ }
};

template<typename T>
class supervisedDataSet
{
public:
	std::vector<supervisedTestData<T>> data;

	void addData(dnnbasic::tensor<T>& in, dnnbasic::tensor<T>& out)
	{
		data.push_back(supervisedTestData(in, out));
	}

	std::size_t size() const
	{
		return data.size();
	}
};

template<typename T>
supervisedDataSet<T> makeSupervisedDatasetIO1x1(std::vector<T> inputs, std::function<T(T)> inputToOutputFunc, const uint32_t batchSize = 1u)
{
	supervisedDataSet<T> dataset;
	for (size_t i = 0; i + batchSize < inputs.size(); i += batchSize)
	{
		std::vector<T> lol;
		std::vector<T> loll;
		for (size_t q = 0; q < batchSize; q++)
		{
			lol.push_back(inputs[i + q]);
			loll.push_back(inputToOutputFunc(inputs[i + q]));
		}

		if (batchSize == 1)
		{
			dnnbasic::tensor<T> in({ 1u }, lol);
			dnnbasic::tensor<T> out({ 1u }, loll);

			dataset.addData(in, out);
		}
		else
		{
			dnnbasic::tensor<T> in({ batchSize, 1u }, lol);
			dnnbasic::tensor<T> out({ batchSize, 1u }, loll);

			dataset.addData(in, out);
		}
	}

	return dataset;
}

int main()
{
	{
		std::vector<float> inputs(100000);
		std::iota(inputs.begin(), inputs.end(), 0);
		for (size_t i = 0; i < inputs.size(); i++)
		{
			inputs[i] = inputs[i] / inputs.size();
		}
		auto dataset = makeSupervisedDatasetIO1x1<float>(inputs, [](auto x) { return x * x + 2; }, 10000);

		auto* opti = new dnnbasic::optimizer::sgd(0.001f);

		dnnbasic::layer::linear<float> l1(1, 10, false);
		dnnbasic::layer::linear<float> l2(10, 1, true);

		dnnbasic::tensor<float> input({ 10000, 1 });
		dnnbasic::tensor<float> output({ 10000, 1 });

		dataset.data[0].input.copyTo(input);
		dataset.data[0].output.copyTo(output);

		dnnbasic::graphRecorder recorder;

		recorder.startRecording();

		auto actual = l2.forward(l1.forward(input));
		auto fisk = dnnbasic::loss::meanSquaredLoss(output, actual, 0);
		fisk.backward(opti);

		recorder.stopRecording();




		for (size_t q = 0; q < 1; q++)
		{
			auto start = std::chrono::system_clock::now();

			for (size_t i = 0; i < 1; i++)
			{

				dataset.data[i].input.copyTo(input);
				dataset.data[i].output.copyTo(output);

				recorder.replay();

				//auto actual = l2.forward(l1.forward(input));
				//auto fisk = dnnbasic::loss::meanSquaredLoss(output, actual, 0);
				//fisk.backward(opti);

				if (i == 0 && q % 10 == 0)
				{
					std::cout << "Error: " << fisk.getError() << std::endl;
				}
			}

			auto end = std::chrono::system_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

			std::cout << "Time: " << time.count() << std::endl;
		}
	}

	cudabasic::resetCudaDevice();
}