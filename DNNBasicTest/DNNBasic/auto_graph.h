#pragma once


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
}