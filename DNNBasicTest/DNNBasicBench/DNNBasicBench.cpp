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
#include "cudaBasics.h"
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

template<typename T>
float benchMarkFunc(const std::function<dnnbasic::tensor<T>(void)>& func)
{
    std::vector<float> benchTimes;
    cudabasic::cudaTimer timer;

    const int32_t benchCount = 20;
    for (size_t i = 0; i < benchCount; i++)
    {
        timer.startTimer();
        auto tmp = func();
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

void benchMarkMatrixMultColumnsVsRows(std::string folder, std::string filename, const uint32_t elementCount)
{
    std::filesystem::create_directory(folder);

    std::string filepath = folder + "/" + filename + ".txt";

    std::vector<matrixSize> matSizes;
    for (uint32_t i = 1; i <= elementCount; i++)
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

    auto randomValues = getVectorWithRandomNumbers(elementCount);

    auto benchSeq = [&](const matrixSize& matSize)
    {
        const std::size_t matElemCount = matSize.columns * matSize.rows;
        std::vector<float> aRange(randomValues.begin(), randomValues.begin() + matElemCount);
        std::vector<float> bRange(randomValues.begin(), randomValues.begin() + matElemCount);

        dnnbasic::tensor<float> a({ matSize.rows, matSize.columns }, aRange);
        dnnbasic::tensor<float> b({ matSize.columns, matSize.rows }, bRange);

        return benchMarkFunc(std::function<dnnbasic::tensor<float>(void)>([=]()
            {
                return a.matMul(b);
            }));
    };

    //Warmup
    benchSeq(matSizes[0]);

    for (size_t i = 0; i < matSizes.size(); i++)
    {
        const matrixSize matSize = matSizes[i];
        float time = benchSeq(matSize);

        const float colRowRatio = (float)matSize.columns / elementCount;
        file << std::to_string(colRowRatio) << "; " << std::to_string(time) << std::endl;

        std::cout << (i + 1) << "/" << matSizes.size() << " " << std::to_string(colRowRatio) << " " << std::to_string(time) << std::endl;
    }
}

int main()
{
    for (uint32_t i = 1024; i <= 1024 * 16; i += 1024)
    {
        benchMarkMatrixMultColumnsVsRows("matmul/static_dynv2_block_size", "matrix_size-" + std::to_string(i), i);
    }

}