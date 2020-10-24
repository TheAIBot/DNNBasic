#pragma once

#include <cstdint>
#include <vector>
#include <variant>
#include <cuda_runtime.h>
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
		void addTensor(tensor<T>& ten);

		void addKernelNode(const cudaKernelNodeParams* kernelParams);
		void addMemsetNode(const cudaMemsetParams* memsetParams);
		void addMemcpyNode(const cudaMemcpy3DParms* memcpyParams);
	};
}