#include <thread>
#include <assert.h>
#include "auto_graph.h"
#include "cudaBasics.h"
#include "cuda_settings.h"

namespace dnnbasic::autoGraph
{
	static thread_local bool makeGraph = true;

	static thread_local bool recordWholeGraph = false;
	static thread_local graphRecorder* currentRecorder = nullptr;

	void setMakeGraph(bool value)
	{
		makeGraph = value;
	}

	bool getMakeGraph()
	{
		return makeGraph;
	}

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

		if (recordWholeGraph)
		{
			assert(currentRecorder != nullptr);

			currentRecorder->addTensor(ten);
			if (node != nullptr)
			{
				auto tensors = node->getTensors();
				for (size_t i = 0; i < tensors.size(); i++)
				{
					currentRecorder->addTensor(tensors[i]);
				}
			}
		}
	}

	template<typename T>
	void forceMakeGraph(tensor<T>& ten, const std::function<tensorNode<T>* ()>& makeTensorNode)
	{
		tensorNode<T>* node = makeTensorNode();
		ten.setNode(node);

		if (recordWholeGraph)
		{
			assert(currentRecorder != nullptr);

			currentRecorder->addTensor(ten);

			auto tensors = node->getTensors();
			for (size_t i = 0; i < tensors.size(); i++)
			{
				currentRecorder->addTensor(tensors[i]);
			}
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
		assert(!recordWholeGraph);
		assert(currentRecorder == nullptr);
		assert(!this->hasRecordedGraph);

		recordWholeGraph = true;
		currentRecorder = this;

		cudaStreamBeginCapture(cuda::getDefaultStream(), cudaStreamCaptureModeGlobal);
	}
	void graphRecorder::stopRecording()
	{
		assert(recordWholeGraph);
		assert(currentRecorder == this);

		recordWholeGraph = false;
		currentRecorder = nullptr;
		this->hasRecordedGraph = true;

		cudaStreamEndCapture(cuda::getDefaultStream(), &this->graph);
		cudaGraphInstantiate(&this->graphExe, this->graph, nullptr, nullptr, 0);
		cudabasic::checkForCudaError();
	}
	void graphRecorder::replay() const
	{
		cudaGraphLaunch(this->graphExe, 0);
		cudabasic::cudaSynchronize();
	}
}