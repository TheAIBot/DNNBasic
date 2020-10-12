#include <memory>
#include <thread>
#include "cuda_settings.h"


namespace dnnbasic::cuda
{
    static thread_local cudabasic::cudaStream* threadStream = nullptr;

    cudaStream_t getDefaultStream()
    {
        if (threadStream == nullptr)
        {
            threadStream = new cudabasic::cudaStream();
        }
        return *threadStream;
    }
}
