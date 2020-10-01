// DNNBasicBench.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <functional>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include <fstream>
#include <random>
#include <filesystem>
#include "cudaTimer.h"
#include "tensor.h"

std::vector<float> getVectorWithRandomNumbers(const uint32_t size)
{
    std::vector<float> numbers;

    std::default_random_engine rngGen(7);
    std::uniform_real_distribution<float> dist(-1322, 64323);
    for (size_t i = 0; i < size; i++)
    {
        numbers.push_back(dist(rngGen));
    }

    return numbers;
}

float benchMarkFunc(std::function<void(void)> func)
{
    std::vector<float> benchTimes;
    cudabasic::cudaTimer timer;

    const int32_t benchCount = 1000;
    for (size_t i = 0; i < benchCount; i++)
    {
        timer.startTimer();
        func();
        timer.stopTimer();
        benchTimes.push_back(timer.getElapsedMiliseconds());
    }

    std::sort(benchTimes.begin(), benchTimes.end());

    const float useOnlyBestTimes = 0.8f;
    const int32_t dwa = (int32_t)(benchCount * useOnlyBestTimes);
    float sum = 0;
    for (size_t i = 0; i < dwa; i++)
    {
        sum += benchTimes[i];
    }

    return sum / dwa;
}

struct matrixSize
{
    uint32_t columns;
    uint32_t rows;

    matrixSize(uint32_t columns, uint32_t rows)
    {
        this->columns = columns;
        this->rows = rows;
    }
};

void benchMarkMatrixMultColumnsVsRows(std::string folder, std::string filename)
{
    std::filesystem::create_directory(folder);

    std::string filepath = folder + "/" + filename;

    const uint32_t elementCount = 1024;

    std::vector<matrixSize> matSizes;
    for (size_t i = 1; i <= elementCount; i++)
    {
        const uint32_t columns = i;
        const uint32_t rows = elementCount / columns;

        if (matSizes.size() == 0 || matSizes.back().rows != rows)
        {
            matSizes.push_back(matrixSize(columns, rows));
        }
        else
        {
            matSizes.back().columns = columns;
            matSizes.back().rows = rows;
        }
    }


    std::ofstream file(filepath);
    file << "cols/rows; time in ms" << std::endl;

    for (size_t i = 0; i < matSizes.size(); i++)
    {
        const matrixSize matSize = matSizes[i];

        auto aData = getVectorWithRandomNumbers(matSize.columns * matSize.rows);
        auto bData = getVectorWithRandomNumbers(matSize.columns * matSize.rows);

        dnnbasic::tensor<float> a({ matSize.rows, matSize.columns }, aData);
        dnnbasic::tensor<float> b({ matSize.columns, matSize.rows }, bData);

        std::vector<dnnbasic::tensor<float>*> resultTensors;
        float time = benchMarkFunc([&]()
            {
                resultTensors.push_back(a.matMul(b));
            });

        for (size_t z = 0; z < resultTensors.size(); z++)
        {
            delete resultTensors[z];
        }

        const float colRowRatio = (float)matSize.columns / elementCount;
        file << std::to_string(colRowRatio) << "; " << std::to_string(time) << std::endl;

        std::cout << (i + 1) << "/" << matSizes.size() << " " << std::to_string(colRowRatio) << " " << std::to_string(time) << std::endl;
    }
}

int main()
{
    benchMarkMatrixMultColumnsVsRows("matmul", "static_block_size.txt");
}