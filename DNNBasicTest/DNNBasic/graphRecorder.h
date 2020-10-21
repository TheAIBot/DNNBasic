#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <variant>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cudaBasics.h"
#include "tensor.h"

namespace dnnbasic
{
	class graphRecorder
	{
	private:
		std::vector<std::variant<
			tensor<bool>,
			tensor<uint8_t>,
			tensor<uint16_t>,
			tensor<uint32_t>,
			tensor<uint64_t>,
			tensor<int8_t>,
			tensor<int16_t>,
			tensor<int32_t>,
			tensor<int64_t>,
			tensor<float>,
			tensor<double>>> tensors;
		cudaGraph_t graph;
		cudaGraphExec_t graphExe;
		bool hasRecordedGraph;
		cudaGraphNode_t prevNode;

	public:
		graphRecorder();
		~graphRecorder();

		void startRecording();
		void stopRecording();
		void replay() const;

		template<typename T>
		void addTensor(tensor<T>& ten)
		{
			tensors.emplace_back(ten);
		}

		void addKernelNode(const cudaKernelNodeParams* kernelParams)
		{
			std::array<cudaGraphNode_t, 1> dependencies = {
				prevNode
			};

			//root node has no dependencies
			const std::size_t depCount = prevNode == nullptr ? 0 : 1;

			cudaGraphNode_t node;
			const cudaError_t status = cudaGraphAddKernelNode(&node, this->graph, &dependencies[0], depCount, kernelParams);
			if (status != cudaError::cudaSuccess)
			{
				cudabasic::checkForCudaError();
			}

			this->prevNode = node;
		}

		void addMemsetNode(const cudaMemsetParams* memsetParams)
		{
			std::array<cudaGraphNode_t, 1> dependencies = {
				prevNode
			};

			//root node has no dependencies
			const std::size_t depCount = prevNode == nullptr ? 0 : 1;

			cudaGraphNode_t node;
			const cudaError_t status = cudaGraphAddMemsetNode(&node, this->graph, &dependencies[0], depCount, memsetParams);
			if (status != cudaError::cudaSuccess)
			{
				cudabasic::checkForCudaError();
			}

			this->prevNode = node;
		}

		void addMemcpyNode(const cudaMemcpy3DParms* memcpyParams)
		{
			std::array<cudaGraphNode_t, 1> dependencies = {
				prevNode
			};

			//root node has no dependencies
			const std::size_t depCount = prevNode == nullptr ? 0 : 1;

			cudaGraphNode_t node;
			const cudaError_t status = cudaGraphAddMemcpyNode(&node, this->graph, &dependencies[0], depCount, memcpyParams);
			if (status != cudaError::cudaSuccess)
			{
				cudabasic::checkForCudaError();
			}

			this->prevNode = node;
		}
	};
}