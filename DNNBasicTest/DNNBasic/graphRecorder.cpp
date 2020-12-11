#include <thread>
#include <array>
#include <stdexcept>
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
		this->prevNode = nullptr;
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
			this->prevNode = nullptr;
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
		cudaGraphLaunch(this->graphExe, cuda::getDefaultStream());
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

	void graphRecorder::addKernelNode(const cudaKernelNodeParams* kernelParams)
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

	void graphRecorder::addMemsetNode(const cudaMemsetParams* memsetParams)
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

	void graphRecorder::addMemcpyNode(const cudaMemcpy3DParms* memcpyParams)
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
}