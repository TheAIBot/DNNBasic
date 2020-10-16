
# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.19)

project ("DNNBasicstuff")

# Include sub-projects.
add_subdirectory ("DNNBasicTest/DNNBasic/")
#add_subdirectory ("/DNNBasicBench/")
#add_subdirectory ("/DNNBasicRun/")
#add_subdirectory ("/DNNBasicTest/")

# TODO: Add tests and install targets if needed.