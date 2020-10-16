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
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../DNNBasic/")

# Add source to this project's executable.
add_executable(dnn_basic_run "DNNBasicRun.cpp")

target_compile_features(dnn_basic_run PUBLIC cxx_std_17)
set_target_properties(dnn_basic_run PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
