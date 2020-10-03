// DNNBasicRun.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Tensor.h"

int main()
{
	dnnbasic::tensor<float> a({ 2, 3, 2 },
				{
					5,7,
					4,8,
					6,1,

					5,7,
					4,8,
					6,1
				});
	dnnbasic::tensor<float> b({ 2, 2, 3 },
				{
					7,5,4,
					7,9,6,

					7,5,4,
					7,9,6
				});

	dnnbasic::tensor<float> expected({ 2, 3, 3 },
				{
					84,88,62,
					84,92,64,
					49,39,30,

					84,88,62,
					84,92,64,
					49,39,30
				});
	auto actual = a.matMul(b);

	std::cout << (actual==expected) << std::endl;
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
