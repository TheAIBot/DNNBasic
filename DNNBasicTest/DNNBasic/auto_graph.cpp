#include <thread>
#include <assert.h>
#include "auto_graph.h"
#include "cudaBasics.h"
#include "cuda_settings.h"
#include "graphRecorder.h"

namespace dnnbasic::autoGraph
{
	static thread_local bool makeGraph = true;

	scopeLevelDisableAutoGraph::scopeLevelDisableAutoGraph()
	{
		this->oldValue = makeGraph;
		makeGraph = false;
	}

	scopeLevelDisableAutoGraph::~scopeLevelDisableAutoGraph()
	{
		makeGraph = this->oldValue;
	}

	template<typename T>
	void handleMakeGraph(tensor<T>& ten, const std::function<tensorNode<T>*()>& makeTensorNode)
	{
		tensorNode<T>* node = nullptr;
		if (makeGraph)
		{
			node = makeTensorNode();
			ten.setNode(node);
		}

		if (isRecordingGraph())
		{
			if (node == nullptr)
			{
				node = makeTensorNode();
			}

			auto graph = getRecordingGraph();

			auto tensors = node->getTensors();
			for (size_t i = 0; i < tensors.size(); i++)
			{
				graph->addTensor(tensors[i]);
			}
			graph->addTensor(ten);
		}
	}

	template<typename T>
	void forceMakeGraph(tensor<T>& ten, const std::function<tensorNode<T>* ()>& makeTensorNode)
	{
		tensorNode<T>* node = makeTensorNode();
		ten.setNode(node);

		if (isRecordingGraph())
		{
			if (node == nullptr)
			{
				node = makeTensorNode();
			}

			auto graph = getRecordingGraph();

			auto tensors = node->getTensors();
			for (size_t i = 0; i < tensors.size(); i++)
			{
				graph->addTensor(tensors[i]);
			}
			graph->addTensor(ten);
		}
	}

	template void handleMakeGraph(tensor<bool>& ten, const std::function<tensorNode<bool>*()>& makeTensorNode);
	template void handleMakeGraph(tensor<uint8_t>& ten, const std::function<tensorNode<uint8_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<uint16_t>& ten, const std::function<tensorNode<uint16_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<uint32_t>& ten, const std::function<tensorNode<uint32_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<uint64_t>& ten, const std::function<tensorNode<uint64_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<int8_t>& ten, const std::function<tensorNode<int8_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<int16_t>& ten, const std::function<tensorNode<int16_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<int32_t>& ten, const std::function<tensorNode<int32_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<int64_t>& ten, const std::function<tensorNode<int64_t>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<float>& ten, const std::function<tensorNode<float>* ()>& makeTensorNode);
	template void handleMakeGraph(tensor<double>& ten, const std::function<tensorNode<double>* ()>& makeTensorNode);

	template void forceMakeGraph(tensor<bool>& ten, const std::function<tensorNode<bool>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<uint8_t>& ten, const std::function<tensorNode<uint8_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<uint16_t>& ten, const std::function<tensorNode<uint16_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<uint32_t>& ten, const std::function<tensorNode<uint32_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<uint64_t>& ten, const std::function<tensorNode<uint64_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<int8_t>& ten, const std::function<tensorNode<int8_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<int16_t>& ten, const std::function<tensorNode<int16_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<int32_t>& ten, const std::function<tensorNode<int32_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<int64_t>& ten, const std::function<tensorNode<int64_t>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<float>& ten, const std::function<tensorNode<float>* ()>& makeTensorNode);
	template void forceMakeGraph(tensor<double>& ten, const std::function<tensorNode<double>* ()>& makeTensorNode);

	static thread_local graphRecorder* currentRecorder = nullptr;

	bool isRecordingGraph()
	{
		return currentRecorder != nullptr;
	}

	graphRecorder* getRecordingGraph()
	{
		return currentRecorder;
	}

	void setGraphRecorder(graphRecorder* recorder)
	{
		currentRecorder = recorder;
	}

	void addNodeToGraph(const std::vector<void*> inputs, const void* output, cudaKernelNodeParams* kernelParams)
	{
		currentRecorder->addKernelNode(inputs, output, kernelParams);
	}

	void addMemsetNodeToGraph(cudaMemsetParams* memsetParams)
	{
		currentRecorder->addMemsetNode(memsetParams);
	}

	void addMemcpyNodeToGraph(cudaMemcpy3DParms* memcpyParams)
	{
		currentRecorder->addMemcpyNode(memcpyParams);
	}
}