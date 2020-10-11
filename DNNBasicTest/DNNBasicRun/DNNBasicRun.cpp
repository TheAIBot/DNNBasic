// DNNBasicRun.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <functional>
#include "tensor.h"
#include "linear.h"
#include "mean_squared_loss.h"
#include "sgd.h"

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
supervisedDataSet<T> makeSupervisedDatasetIO1x1(std::vector<T> inputs, std::function<T(T)> inputToOutputFunc)
{
	supervisedDataSet<T> dataset;
	for (size_t i = 0; i < inputs.size(); i++)
	{
		std::vector<T> lol = { inputs[i] };
		std::vector<T> loll = { inputToOutputFunc(inputs[i]) };
		dnnbasic::tensor<T> in({ 1u }, lol);
		dnnbasic::tensor<T> out({ 1u }, loll);

		dataset.addData(in, out);
	}

	return dataset;
}

int main()
{
	std::vector<float> inputs(100);
	std::iota(inputs.begin(), inputs.end(), 0);
	auto dataset = makeSupervisedDatasetIO1x1<float>(inputs, [](auto x) { return x; });

	auto* opti = new dnnbasic::optimizer::sgd(0.001);

	dnnbasic::layer::linear<float> l1(1, 1, false);
	for (size_t q = 0; q < 30; q++)
	{
		for (size_t i = 0; i < dataset.size(); i++)
		{
			auto expected = dataset.data[i].output;
			auto actual = l1.forward(dataset.data[i].input);

			auto fisk = dnnbasic::loss::meanSquaredLoss(expected, actual);
			//std::cout << std::abs(actual.getValuesOnCPU()[0] - expected.getValuesOnCPU()[0]) << std::endl;
			fisk.backward(opti);
		}
	}
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
