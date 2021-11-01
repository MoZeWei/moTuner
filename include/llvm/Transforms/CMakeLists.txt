cmake_minimum_required(VERSION 3.1)
project(Mark_Dimension_FuncID)
project(SbS_Tuning)
project(data_flow)

# support C++14 features used by LLVM 10.0.0
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "-Wall -fno-rtti")
find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(/mnt/data/home/mzw/hipcc_llvm_4.3.0/include)
link_directories(/mnt/data/home/mzw/hipcc_llvm_4.3.0/lib)

#add_subdirectory(skeleton)  # Use your pass name here.
add_subdirectory(Mark_Dimension_FuncID)
add_subdirectory(SbS_Tuning)
add_subdirectory(data_flow)
