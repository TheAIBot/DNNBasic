#pragma once


namespace dnnbasic::autoGraph
{
	void setMakeGraph(bool value);
	bool getMakeGraph();

	class scopeLevelDisableAutoGrad
	{
	private:
		bool oldValue;

	public:
		scopeLevelDisableAutoGrad();
		~scopeLevelDisableAutoGrad();
	};
}