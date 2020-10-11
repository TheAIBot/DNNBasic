#include <thread>
#include "auto_graph.h"

namespace dnnbasic::autoGraph
{
	static thread_local bool makeGraph = true;

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
}