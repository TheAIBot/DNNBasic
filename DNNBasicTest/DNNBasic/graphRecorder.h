#pragma once

#include <cstdint>
#include <vector>
#include <variant>
#include <cuda_runtime.h>
#include <unordered_map>
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
		std::unordered_map<const void*, cudaGraphNode_t> currNodeDeps;

		std::vector<cudaGraphNode_t> getDepNodes(const std::vector<void*>& inputs) const;
		std::vector<cudaGraphNode_t> getDepNodes(const std::vector<const void*>& inputs) const;

	public:
		graphRecorder();
		~graphRecorder();

		void startRecording();
		void stopRecording();
		void replay() const;

		template<typename T>
		void addTensor(tensor<T>& ten);

		void addKernelNode(const std::vector<void*>& inputs, const void* output, const cudaKernelNodeParams* kernelParams);
		void addMemsetNode(const void* input, const void* output, const cudaMemsetParams* memsetParams);
		void addMemcpyNode(const void* input, const void* output, const cudaMemcpy3DParms* memcpyParams);
	};
}