#include <thread>
#include <array>
#include <stdexcept>
#include <algorithm>
#include "graphRecorder.h"
#include "auto_graph.h"
#include "cuda_settings.h"
#include "cudaBasics.h"

namespace dnnbasic
{
	graphRecorder::graphRecorder()
	{
		cudaGraphCreate(&this->graph, 0);
		this->graphExe = nullptr;
		this->hasRecordedGraph = false;
	}
	graphRecorder::~graphRecorder()
	{
		cudaGraphDestroy(this->graph);
		if (this->hasRecordedGraph)
		{
			cudaGraphExecDestroy(this->graphExe);
		}
	}

	void graphRecorder::startRecording()
	{
		if (autoGraph::isRecordingGraph())
		{
			throw std::runtime_error("Two graphs can't record at the same time.");
		}

		//in case the graph recorder is reused for recording
		if (this->hasRecordedGraph)
		{
			//destroy and make new graphs
			cudaGraphDestroy(this->graph);
			cudaGraphExecDestroy(this->graphExe);

			cudaGraphCreate(&this->graph, 0);
			this->graphExe = nullptr;
			this->hasRecordedGraph = false;
		}

		autoGraph::setGraphRecorder(this);
	}
	void graphRecorder::stopRecording()
	{
		if (!autoGraph::isRecordingGraph())
		{
			throw std::runtime_error("Can't stop recording because nothing was recording.");
		}
		if (autoGraph::getRecordingGraph() != this)
		{
			throw std::runtime_error("This recorder wasn't the one that started recording.");
		}

		autoGraph::setGraphRecorder(nullptr);
		this->hasRecordedGraph = true;

		cudaGraphInstantiate(&this->graphExe, this->graph, nullptr, nullptr, 0);
		cudabasic::checkForCudaError();
	}
	void graphRecorder::replay() const
	{
		cudaGraphLaunch(this->graphExe, 0);
		cudaStreamSynchronize(cuda::getDefaultStream());
	}

	template<typename T>
	void graphRecorder::addTensor(tensor<T>& ten)
	{
		tensors.emplace_back(ten);
	}

	template void graphRecorder::addTensor(tensor<bool>& ten);
	template void graphRecorder::addTensor(tensor<uint8_t>& ten);
	template void graphRecorder::addTensor(tensor<uint16_t>& ten);
	template void graphRecorder::addTensor(tensor<uint32_t>& ten);
	template void graphRecorder::addTensor(tensor<uint64_t>& ten);
	template void graphRecorder::addTensor(tensor<int8_t>& ten);
	template void graphRecorder::addTensor(tensor<int16_t>& ten);
	template void graphRecorder::addTensor(tensor<int32_t>& ten);
	template void graphRecorder::addTensor(tensor<int64_t>& ten);
	template void graphRecorder::addTensor(tensor<float>& ten);
	template void graphRecorder::addTensor(tensor<double>& ten);

	std::vector<cudaGraphNode_t> graphRecorder::getDepNodes(const std::vector<void*>& inputs) const
	{
		const std::vector<const void*> cInputs(inputs.begin(), inputs.end());
		return getDepNodes(cInputs);
	}

	std::vector<cudaGraphNode_t> graphRecorder::getDepNodes(const std::vector<const void*>& inputs) const
	{
		std::vector<cudaGraphNode_t> depNodes;
		for (size_t i = 0; i < inputs.size(); i++)
		{
			auto val = this->currNodeDeps.find(inputs[i]);
			if (val != this->currNodeDeps.end())
			{
				//can't have duplicate dependicies
				if (std::find(depNodes.begin(), depNodes.end(), val->second) == depNodes.end())
				{
					depNodes.push_back(val->second);
				}
			}
		}

		return depNodes;
	}

	void graphRecorder::addKernelNode(const std::vector<void*>& inputs, const void* output, const cudaKernelNodeParams* kernelParams)
	{
		std::vector<cudaGraphNode_t> depNodes = this->getDepNodes(inputs);

		cudaGraphNode_t node;
		const cudaError_t status = cudaGraphAddKernelNode(&node, this->graph, depNodes.data(), depNodes.size(), kernelParams);
		if (status != cudaError::cudaSuccess)
		{
			cudabasic::checkForCudaError();
		}

		this->currNodeDeps.insert_or_assign(output, node);
	}

	void graphRecorder::addMemsetNode(const void* input, const void* output, const cudaMemsetParams* memsetParams)
	{
		std::vector<cudaGraphNode_t> depNodes = this->getDepNodes({ input });

		cudaGraphNode_t node;
		const cudaError_t status = cudaGraphAddMemsetNode(&node, this->graph, depNodes.data(), depNodes.size(), memsetParams);
		if (status != cudaError::cudaSuccess)
		{
			cudabasic::checkForCudaError();
		}

		this->currNodeDeps.insert_or_assign(output, node);
	}

	void graphRecorder::addMemcpyNode(const void* input, const void* output, const cudaMemcpy3DParms* memcpyParams)
	{
		std::vector<cudaGraphNode_t> depNodes = this->getDepNodes({ input });

		cudaGraphNode_t node;
		const cudaError_t status = cudaGraphAddMemcpyNode(&node, this->graph, depNodes.data(), depNodes.size(), memcpyParams);
		if (status != cudaError::cudaSuccess)
		{
			cudabasic::checkForCudaError();
		}

		this->currNodeDeps.insert_or_assign(output, node);
	}
}