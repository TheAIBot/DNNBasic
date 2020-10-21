#pragma once

#include <functional>
#include "tensor.h"
#include "span.h"

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
		void addMemsetNodeToGraph(cudaMemsetParams* memsetParams);
		void addMemcpyNodeToGraph(cudaMemcpy3DParms* memcpyParams);

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
			kernelParams.func = reinterpret_cast<void*>(kernel);
			kernelParams.blockDim = blockDim;
			kernelParams.gridDim = gridDim;
			kernelParams.sharedMemBytes = sharedMemSize;
			kernelParams.kernelParams = &arguments[0];
			kernelParams.extra = nullptr;

			addNodeToGraph(&kernelParams);
		}

		template<typename T>
		void addMemsetNode(cudabasic::span<T> dst, uint32_t value)
		{
			if (!isRecordingGraph())
			{
				throw std::runtime_error("Tried to record memset node without starting to record first.");
			}

			cudaMemsetParams memsetParams;
			memsetParams.dst = reinterpret_cast<void*>(dst.begin());
			memsetParams.elementSize = 1;
			memsetParams.height = 1;
			memsetParams.pitch = 1;
			memsetParams.value = value;
			memsetParams.width = dst.size() * sizeof(T);

			addMemsetNodeToGraph(&memsetParams);
		}

		template<typename T>
		void addMemcpyNode(cudabasic::span<T> from, cudabasic::span<T> to)
		{
			if (!isRecordingGraph())
			{
				throw std::runtime_error("Tried to record memcpy node without starting to record first.");
			}

			if (from.size() != to.size())
			{
				throw std::runtime_error("The size of the span that is copied from must be the same as the one that is copied to.");
			}

			cudaMemcpy3DParms memcpyParams = { 0 };
			memcpyParams.dstPos = make_cudaPos(0, 0, 0);
			memcpyParams.dstPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(to.begin()), to.size() * sizeof(T), to.size(), 1);
			memcpyParams.extent = make_cudaExtent(from.size() * sizeof(T), 1, 1);
			memcpyParams.kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
			memcpyParams.srcPos = make_cudaPos(0, 0, 0);
			memcpyParams.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(from.begin()), from.size() * sizeof(T), from.size(), 1);

			addMemcpyNodeToGraph(&memcpyParams);
		}
	}
}