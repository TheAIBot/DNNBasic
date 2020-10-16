#include <stdexcept>
#include <cstdint>
#include "sgd.h"
#include "tensor.h"
#include "tensor_elementwise_kernels.cuh"

namespace dnnbasic::optimizer
{
    template<typename T>
    void sgd::update(tensor<T>& weights, const tensor<T>& gradients)
    {
        auto lrGrad = (gradients.cast<float>() * this->learningRate).cast<T>();
        tensorSubtract(weights, lrGrad, weights, false);
    }

    sgd::sgd(float learningRate) : learningRate(learningRate)
    { }

    void sgd::updateWeights(tensor<bool>& weights, const tensor<bool>& gradients)
    {
        throw std::runtime_error("Can't update weights for bool tensors yet.");
    }

    void sgd::updateWeights(tensor<uint8_t>& weights, const tensor<uint8_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<uint16_t>& weights, const tensor<uint16_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<uint32_t>& weights, const tensor<uint32_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<uint64_t>& weights, const tensor<uint64_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<int8_t>& weights, const tensor<int8_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<int16_t>& weights, const tensor<int16_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<int32_t>& weights, const tensor<int32_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<int64_t>& weights, const tensor<int64_t>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<float>& weights, const tensor<float>& gradients) { update(weights, gradients); }
    void sgd::updateWeights(tensor<double>& weights, const tensor<double>& gradients) { update(weights, gradients); }
}