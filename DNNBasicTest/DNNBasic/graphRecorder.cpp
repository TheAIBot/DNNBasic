#include <thread>
#include "graphRecorder.h"
#include "auto_graph.h"
#include "cuda_settings.h"

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
		cudaGraphLaunch(this->graphExe, 0);
		cudaStreamSynchronize(cuda::getDefaultStream());
	}
}