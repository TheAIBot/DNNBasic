#pragma once

#include <functional>
#include "tensor.h"

namespace dnnbasic
{
	class graphRecorder;
}

namespace dnnbasic
{
	namespace autoGraph
	{
		class scopeLevelDisableAutoGraph
		{
		private:
			bool oldValue;

		public:
			scopeLevelDisableAutoGraph();
			~scopeLevelDisableAutoGraph();
		};

		template<typename T>
		void handleMakeGraph(tensor<T>& ten, const std::function<tensorNode<T>* ()>& makeTensorNode);

		template<typename T>
		void forceMakeGraph(tensor<T>& ten, const std::function<tensorNode<T>* ()>& makeTensorNode);


		bool isRecordingGraph();
		graphRecorder* getRecordingGraph();
		void setGraphRecorder(graphRecorder* recorder);
		void addNodeToGraph(cudaKernelNodeParams* kernelParams);

		template<typename... Args>
		void addKernelNode(void(*kernel)(Args...), dim3 blockDim, dim3 gridDim, size_t sharedMemSize, Args... args)
		{
			if (!isRecordingGraph())
			{
				throw std::runtime_error("Tried to record kernel node without starting to record first.");
			}

			std::array<void*, sizeof...(args)> arguments = {
				&args...
			};

			cudaKernelNodeParams kernelParams;
			kernelParams.func = kernel;
			kernelParams.blockDim = blockDim;
			kernelParams.gridDim = gridDim;
			kernelParams.sharedMemBytes = sharedMemSize;
			kernelParams.kernelParams = &arguments[0];
			kernelParams.extra = nullptr;

			addNodeToGraph(&kernelParams);
		}
	}
}