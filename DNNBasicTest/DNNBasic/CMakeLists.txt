# CMakeList.txt : CMake project for ISATestsGen, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.19)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "35")

project(DNNBasic LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../CudaBasicClasses/")

# Add source to this project's executable.
add_library(dnn_basic 
    "kernel_tools.h"
    "kernel_tools.cpp"
    "tensor.h"
    "tensor.cpp"
    "tensor_multi_dim_matrix_mul.cu"
    "tensor_multi_dim_matrix_mul.cuh"
    "tensor_cast_kernel.cuh"
    "tensor_cast_kernel.cu"
    "tensor_cast.cpp"
    "tensor_permute_kernel.cuh"
    "tensor_permute_kernel.cu"
    "tensor_permute.cpp"
    "tensor_matrix_kernels.cu"
    "tensor_matrix_kernels.cuh"
    "tensor_matrixMultiply.cpp"
    "tensor_sum_kernel.cuh"
    "tensor_sum_kernel.cu"
    "tensor_sum.cpp"
    "tensor_elementwise_kernels.cuh"
    "tensor_elementwise_kernels.cu"
    "tensor_basic_math_operators.cpp"
    "tensor_data.h"
    "tensor_data.cpp"
    "tensor_broadcast.h"
    "tensor_node.h"
    "tensor_node_linear.h"
    "tensor_node_no_grad.h"
    "linear.h"
    "linear.cpp"
    "FBPropagation.h"
    "matrix.h"
    "Functions.h"
    "mean_squared_loss.h"
    "mean_squared_loss.cpp"
    "auto_graph.h"
    "auto_graph.cpp"
    "optimizer.h"
    "cuda_settings.h"
    "cuda_settings.cpp"
    "optional.h"
    "random.h"
    "random.cpp"
    "graphRecorder.h"
    "graphRecorder.cpp"
    "sgd.h"
)

target_compile_features(dnn_basic PUBLIC cxx_std_17)
set_target_properties(dnn_basic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
