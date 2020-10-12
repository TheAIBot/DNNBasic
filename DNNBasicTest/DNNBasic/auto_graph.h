#pragma once

#include <vector>
#include <variant>
#include <functional>
#include <cuda_runtime.h>
#include "tensor.h"

//namespace dnnbasic
//{
//	template<typename T>
//	class tensor;
//
//	template<typename T>
//	class tensorNode;
//}

namespace dnnbasic::autoGraph
{
	void setMakeGraph(bool value);
	bool getMakeGraph();

	class scopeLevelDisableAutoGraph
	{
	private:
		bool oldValue;

	public:
		scopeLevelDisableAutoGraph();
		~scopeLevelDisableAutoGraph();
	};

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

	public:
		template<typename T>
		void addTensor(tensor<T>& ten)
		{
			tensors.emplace_back(ten);
		}

		void startRecording();
		void stopRecording();
		void replay() const;
	};

	template<typename T>
	void handleMakeGraph(tensor<T>& ten, const std::function<tensorNode<T>* ()>& makeTensorNode);

	template<typename T>
	void forceMakeGraph(tensor<T>& ten, const std::function<tensorNode<T>* ()>& makeTensorNode);
}